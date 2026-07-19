import argparse
import os
import random
import re
import warnings
import glob
from pathlib import Path
from collections import OrderedDict
from time import time

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from loguru import logger
import motmetrics

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator

def build_argument_parser():
    parser = argparse.ArgumentParser("AeroTrack Evaluation Engine")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--dist-url", default=None, type=str)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-d", "--devices", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--machine_rank", type=int, default=0)
    parser.add_argument("-f", "--exp_file", default=None, type=str, required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--trt", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--speed", action="store_true")
    parser.add_argument("-c", "--ckpt", default=None, type=str, required=True)
    parser.add_argument("--conf", default=0.01, type=float)
    parser.add_argument("--nms", default=0.69, type=float)
    parser.add_argument("--tsize", default=640, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--track_thresh", type=float, default=0.6)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.9)
    parser.add_argument("--min-box-area", type=float, default=100)
    parser.add_argument("--mot20", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser

def extract_base_gflops(model_info_string):
    match = re.search(r"Gflops:\s*([0-9]+(?:\.[0-9]+)?)", model_info_string)
    if match is None:
        return None
    return float(match.group(1))

import json
import cv2
import numpy as np

def ensure_coco_json_exists(data_dir, json_filename, is_mot20=False):
    """
    Vérifie si le JSON COCO existe et contient des annotations. 
    Sinon, le génère à partir des dossiers de séquences MOT.
    """
    annotations_dir = os.path.join(data_dir, "annotations")
    json_path = os.path.join(annotations_dir, json_filename)
    
    # Vérification rapide : si le fichier existe et a des annotations, on passe.
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if 'annotations' in data and len(data['annotations']) > 0:
                    logger.info(f"Fichier JSON valide trouvé : {json_path}")
                    return
        except Exception:
            pass # Si le fichier est corrompu, on le recrée
            
    logger.info(f"Génération automatique du fichier JSON COCO : {json_path}")
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Déterminer le dossier source (souvent 'test' ou 'train' selon tes splits)
    # Par défaut, on cherche dans 'train' si c'est pour l'évaluation standard
    split_folder = "test"
    source_dir = os.path.join(data_dir, split_folder)
    
    out = {
        'images': [], 
        'annotations': [], 
        'videos': [],
        'categories': [{'id': 1, 'name': 'UAV'}] # Ta catégorie
    }
    
    if not os.path.exists(source_dir):
        logger.error(f"Le dossier source {source_dir} est introuvable !")
        return

    seqs = os.listdir(source_dir)
    image_cnt, ann_cnt, video_cnt, tid_curr, tid_last = 0, 0, 0, 0, -1

    for seq in sorted(seqs):
        if '.DS_Store' in seq: continue
            
        video_cnt += 1
        out['videos'].append({'id': video_cnt, 'file_name': seq})
        seq_path = os.path.join(source_dir, seq)
        img_path = os.path.join(seq_path, 'img1')
        ann_path = os.path.join(seq_path, 'gt/gt.txt')
        
        images = os.listdir(img_path)
        num_images = len([img for img in images if 'jpg' in img])

        # 1. Traitement des images
        for i in range(num_images):
            img_file = os.path.join(img_path, f'{i + 1:06d}.jpg')
            img = cv2.imread(img_file)
            if img is None: continue
                
            height, width = img.shape[:2]
            out['images'].append({
                'file_name': f'{seq}/img1/{i + 1:06d}.jpg',
                'id': image_cnt + i + 1,
                'frame_id': i + 1,
                'prev_image_id': image_cnt + i if i > 0 else -1,
                'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                'video_id': video_cnt,
                'height': height, 
                'width': width
            })
            
        # 2. Traitement des annotations
        if os.path.exists(ann_path):
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
            # Gestion du cas où le fichier gt.txt ne contient qu'une seule ligne
            if anns.ndim == 1: anns = np.expand_dims(anns, axis=0)
                
            for i in range(anns.shape[0]):
                frame_id = int(anns[i][0])
                track_id = int(anns[i][1])
                ann_cnt += 1
                
                if not track_id == tid_last:
                    tid_curr += 1
                    tid_last = track_id
                    
                out['annotations'].append({
                    'id': ann_cnt,
                    'category_id': 1,
                    'image_id': image_cnt + frame_id,
                    'track_id': tid_curr,
                    'bbox': anns[i][2:6].tolist(),
                    'conf': float(anns[i][6]) if anns.shape[1] > 6 else 1.0,
                    'iscrowd': 0,
                    'area': float(anns[i][4] * anns[i][5])
                })
                
        image_cnt += num_images
        
    with open(json_path, 'w') as f:
        json.dump(out, f)
    logger.info(f"Conversion terminée avec {len(out['images'])} images et {len(out['annotations'])} annotations.")

def calculate_effective_computation(early_exit_count, total_frames, base_gflops, shallow_gflops_ratio=0.40):
    if total_frames <= 0 or base_gflops is None:
        return None, None, None
        
    early_exit_rate = early_exit_count / total_frames
    deep_routing_rate = 1.0 - early_exit_rate
    
    shallow_gflops = base_gflops * shallow_gflops_ratio
    effective_gflops = (early_exit_rate * shallow_gflops) + (deep_routing_rate * base_gflops)
    gflops_reduction_percentage = (1.0 - (effective_gflops / base_gflops)) * 100.0
    return effective_gflops, gflops_reduction_percentage, early_exit_rate


def compute_tracking_metrics(ground_truth_dataframes, test_dataframes):
    accuracies = []
    sequence_names = []
    for sequence_key, test_dataframe in test_dataframes.items():
        if sequence_key in ground_truth_dataframes:
            accuracies.append(motmetrics.utils.compare_to_groundtruth(ground_truth_dataframes[sequence_key], test_dataframe, 'iou', distth=0.5))
            sequence_names.append(sequence_key)
    return accuracies, sequence_names

@logger.catch
def execute_evaluation(experiment_configuration, execution_arguments, gpu_count):
    if execution_arguments.seed is not None:
        random.seed(execution_arguments.seed)
        torch.manual_seed(execution_arguments.seed)
        cudnn.deterministic = True

    if hasattr(experiment_configuration, 'tracking_distance_metric'):
        execution_arguments.distance = experiment_configuration.tracking_distance_metric

    is_distributed_execution = gpu_count > 1
    cudnn.benchmark = True
    computation_rank = execution_arguments.local_rank

    output_directory_path = os.path.join(experiment_configuration.output_dir, execution_arguments.experiment_name)
    if computation_rank == 0:
        os.makedirs(output_directory_path, exist_ok=True)

    tracking_results_directory = os.path.join(output_directory_path, "track_results")
    os.makedirs(tracking_results_directory, exist_ok=True)

    setup_logger(output_directory_path, distributed_rank=computation_rank, filename="val_log.txt", mode="a")

    if execution_arguments.conf is not None:
        experiment_configuration.test_conf = execution_arguments.conf
    if execution_arguments.nms is not None:
        experiment_configuration.nmsthre = execution_arguments.nms
    if execution_arguments.tsize is not None:
        experiment_configuration.test_size = (execution_arguments.tsize, execution_arguments.tsize)

    detection_model = experiment_configuration.get_model()
    model_architecture_info = get_model_info(detection_model, experiment_configuration.test_size)
    logger.info(f"Model Summary: {model_architecture_info}")

    base_gflops = getattr(experiment_configuration, "full_network_gflops", extract_base_gflops(model_architecture_info))

    validation_data_loader = experiment_configuration.get_eval_loader(execution_arguments.batch_size, is_distributed_execution, execution_arguments.test)
    
    tracking_evaluator = MOTEvaluator(
        args=execution_arguments,
        dataloader=validation_data_loader,
        img_size=experiment_configuration.test_size,
        confthre=experiment_configuration.test_conf,
        nmsthre=experiment_configuration.nmsthre,
        num_classes=experiment_configuration.num_classes,
    )

    torch.cuda.set_device(computation_rank)
    detection_model.cuda(computation_rank)
    detection_model.eval()

    if not execution_arguments.speed and not execution_arguments.trt:
        checkpoint_payload = torch.load(execution_arguments.ckpt, map_location=f"cuda:{computation_rank}")
        detection_model.load_state_dict(checkpoint_payload.get("model_state_dict", checkpoint_payload.get("model")))

    if is_distributed_execution:
        detection_model = DistributedDataParallel(detection_model, device_ids=[computation_rank])

    if execution_arguments.fuse:
        detection_model = fuse_model(detection_model)

    logger.info("Starting performance evaluation...")
    start_time = time()
    *_, evaluation_summary = tracking_evaluator.evaluate(
        detection_model, is_distributed_execution, execution_arguments.fp16, None, None, experiment_configuration.test_size, tracking_results_directory
    )
    end_time = time()
    
    total_execution_time = end_time - start_time
    total_processed_frames = len(validation_data_loader.dataset)
    latency_milliseconds = (total_execution_time / max(total_processed_frames, 1)) * 1000.0
    frames_per_second = total_processed_frames / max(total_execution_time, 1e-9)

    logger.info(f"\n{evaluation_summary}")

    if hasattr(tracking_evaluator, "last_early_stats") or hasattr(detection_model, "early_exit_enabled"):
        early_exit_statistics = getattr(tracking_evaluator, "last_early_stats", {})
        aggregate_early_exits = sum(stats["early"] for stats in early_exit_statistics.values())
        aggregate_total_frames = sum(stats["total"] for stats in early_exit_statistics.values())
        
        if aggregate_total_frames > 0:
            effective_gflops, gflops_reduction, exit_rate = calculate_effective_computation(
                aggregate_early_exits, aggregate_total_frames, base_gflops
            )
            logger.info(f"Early Exits: {aggregate_early_exits}/{aggregate_total_frames} ({exit_rate:.1%}) | Effective GFLOPs: {effective_gflops:.2f} | GFLOPs Reduction: {gflops_reduction:.2f}%")


    motmetrics.lap.default_solver = 'lap'
    ground_truth_type_suffix = '_val_half' if experiment_configuration.val_ann == 'val_half.json' else ''
    
    # On construit le motif de recherche (le pattern)
    split_folder = 'test'
    gt_pattern = os.path.join(experiment_configuration.data_dir, split_folder, f'*/gt/gt{ground_truth_type_suffix}.txt')
    
    ground_truth_files = glob.glob(gt_pattern)
    test_result_files = [file_path for file_path in glob.glob(os.path.join(tracking_results_directory, '*.txt')) if not os.path.basename(file_path).startswith('eval')]

    # --- DEBUG : Vérifions ce que le script trouve sur ton disque ---
    logger.info(f"Recherche des fichiers GT avec le chemin : {gt_pattern}")
    logger.info(f"Nombre de fichiers GT trouvés : {len(ground_truth_files)}")
    logger.info(f"Nombre de fichiers de résultats trouvés : {len(test_result_files)}")
    # ----------------------------------------------------------------

    # 2. Chargement des données dans les DataFrames
    ground_truth_dataframes = OrderedDict([(Path(file_path).parts[-3], motmetrics.io.loadtxt(file_path, fmt='mot15-2D', min_confidence=1)) for file_path in ground_truth_files])
    test_result_dataframes = OrderedDict([(os.path.splitext(Path(file_path).parts[-1])[0], motmetrics.io.loadtxt(file_path, fmt='mot15-2D', min_confidence=-1)) for file_path in test_result_files])

    # --- DEBUG : Vérifions la correspondance des noms de séquences ---
    logger.info(f"Clés des séquences GT : {list(ground_truth_dataframes.keys())}")
    logger.info(f"Clés des séquences Test : {list(test_result_dataframes.keys())}")
    # -----------------------------------------------------------------

    # 3. Calcul des métriques
    metrics_host = motmetrics.metrics.create()
    computed_accuracies, sequence_names = compute_tracking_metrics(ground_truth_dataframes, test_result_dataframes)
    target_metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    final_metrics_summary = metrics_host.compute_many(computed_accuracies, names=sequence_names, metrics=target_metrics, generate_overall=True)

    ratio_divisors = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']
    }
    
    for divisor_key, dividend_keys in ratio_divisors.items():
        for dividend_key in dividend_keys:
            final_metrics_summary[dividend_key] = (final_metrics_summary[dividend_key] / final_metrics_summary[divisor_key])
            
    final_metrics_summary.loc['OVERALL', 'Latency_ms'] = round(latency_milliseconds, 2)
    final_metrics_summary.loc['OVERALL', 'FPS'] = round(frames_per_second, 2)
    
    metric_formatters = metrics_host.formatters
    percentage_format_keys = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked', 'partially_tracked', 'mostly_lost']
    for format_key in percentage_format_keys:
        metric_formatters[format_key] = metric_formatters['mota']

    print(motmetrics.io.render_summary(final_metrics_summary, formatters=metric_formatters, namemap=motmetrics.io.motchallenge_metric_names))

    csv_output_path = os.path.join(output_directory_path, "mot_evaluation_metrics.csv")
    final_metrics_summary.to_csv(csv_output_path)
    
    logger.info(f"End-to-End Speed -> Latency: {latency_milliseconds:.2f} ms/img | FPS: {frames_per_second:.2f}")

if __name__ == "__main__":
    parsed_arguments = build_argument_parser().parse_args()
    active_experiment = get_exp(parsed_arguments.exp_file, parsed_arguments.name)
    active_experiment.merge(parsed_arguments.opts)

    if not parsed_arguments.experiment_name:
        parsed_arguments.experiment_name = active_experiment.exp_name

    available_gpus = torch.cuda.device_count() if parsed_arguments.devices is None else parsed_arguments.devices
    launch(
        execute_evaluation,
        available_gpus,
        parsed_arguments.num_machines,
        parsed_arguments.machine_rank,
        backend=parsed_arguments.dist_backend,
        dist_url=parsed_arguments.dist_url,
        args=(active_experiment, parsed_arguments, available_gpus),
    )