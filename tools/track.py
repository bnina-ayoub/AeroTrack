import argparse
import os
import random
import re
import sys
import numpy as np
import warnings
import glob
import cv2
import json
import motmetrics as mm
from time import time
from pathlib import Path
from collections import OrderedDict, defaultdict
from thop import profile
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger

# -----------------------------------------------------------------------------
# 1. Environment & Path Configuration
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # Priority to custom aerotrack package

# Submodule path for baseline ByteTrack imports
BYTETRACK_SUBMODULE = os.path.join(ROOT, "third_party", "ByteTrack")
if os.path.exists(BYTETRACK_SUBMODULE) and BYTETRACK_SUBMODULE not in sys.path:
    sys.path.append(BYTETRACK_SUBMODULE)

# -----------------------------------------------------------------------------
# 2. AeroTrack Imports
# -----------------------------------------------------------------------------
from aerotrack.core import launch
from aerotrack.exp import get_exp
from aerotrack.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from aerotrack.evaluators import MOTEvaluator
from aerotrack.evaluators.mot_evaluator import summarize_frame_latency_records
from aerotrack.utils.visualize import plot_tracking


def make_parser():
    parser = argparse.ArgumentParser("AeroTrack Evaluation & Tracking")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your experiment description file")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")
    parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.")
    parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.69, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=50, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.97, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=50, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--distance", type=str, default="nwd", choices=["nwd", "iou"], help="distance metric for tracking")
    parser.add_argument("--save_vis", dest="save_vis", default=False, action="store_true", help="save per-frame tracking visualizations")
    parser.add_argument("--vis_output", type=str, default=None, help="output directory for visualized tracking frames")
    parser.add_argument("--vis_video", dest="vis_video", default=False, action="store_true", help="also export one mp4 per sequence")
    parser.add_argument("--vis_fps", type=int, default=30, help="fps used for exported visualization videos")
    parser.add_argument("--early_exit", dest="early_exit", default=False, action="store_true", help="Activer le routage dynamique Early Exit.")
    return parser


def compare_dataframes(gts, ts):
    accs = []
    names = []
    logger.info('Comparing groundtruth and test dataframes.')
    for k, tsacc in ts.items():
        if k in gts:            
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))
    return accs, names


def parse_gflops_from_model_info(model_info):
    match = re.search(r"Gflops:\s*([0-9]+(?:\.[0-9]+)?)", model_info)
    if match is None:
        return None
    return float(match.group(1))


def compute_effective_gflops(early_count, total_count, full_network_gflops, p3_only_gflops):
    if total_count <= 0:
        raise ValueError("total_count doit être supérieur à zéro.")
    if full_network_gflops is None or p3_only_gflops is None:
        raise ValueError("Les valeurs GFLOPs (full et P3) sont requises pour une mesure exacte.")

    exit_rate = early_count / total_count
    deep_rate = 1.0 - exit_rate
    effective_gflops = (exit_rate * p3_only_gflops) + (deep_rate * full_network_gflops)
    gflops_reduction = (1.0 - (effective_gflops / full_network_gflops)) * 100.0
    
    return effective_gflops, gflops_reduction, exit_rate


def ensure_coco_json_exists(data_dir, json_filename, is_mot20=False):
    annotations_dir = os.path.join(data_dir, "annotations")
    json_path = os.path.join(annotations_dir, json_filename)
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if 'annotations' in data and len(data['annotations']) > 0:
                    logger.info(f"Fichier JSON valide trouvé : {json_path}")
                    return
        except Exception:
            pass 
            
    logger.info(f"Génération automatique du JSON COCO : {json_path}")
    os.makedirs(annotations_dir, exist_ok=True)
    
    split_folder = "test" 
    source_dir = os.path.join(data_dir, split_folder)
    
    out = {'images': [], 'annotations': [], 'videos': [], 'categories': [{'id': 1, 'name': 'UAV'}]}
    
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

        for i in range(num_images):
            img_file = os.path.join(img_path, f'{i + 1:06d}.jpg')
            img = cv2.imread(img_file)
            if img is None: continue
                
            height, width = img.shape[:2]
            out['images'].append({
                'file_name': f'{seq}/img1/{i + 1:06d}.jpg', 'id': image_cnt + i + 1, 'frame_id': i + 1,
                'prev_image_id': image_cnt + i if i > 0 else -1,
                'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                'video_id': video_cnt, 'height': height, 'width': width
            })
            
        if os.path.exists(ann_path):
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
            if anns.ndim == 1: anns = np.expand_dims(anns, axis=0)
            for i in range(anns.shape[0]):
                frame_id = int(anns[i][0])
                track_id = int(anns[i][1])
                ann_cnt += 1
                if not track_id == tid_last:
                    tid_curr += 1
                    tid_last = track_id
                out['annotations'].append({
                    'id': ann_cnt, 'category_id': 1, 'image_id': image_cnt + frame_id, 'track_id': tid_curr,
                    'bbox': anns[i][2:6].tolist(), 'conf': float(anns[i][6]) if anns.shape[1] > 6 else 1.0,
                    'iscrowd': 0, 'area': float(anns[i][4] * anns[i][5])
                })
        image_cnt += num_images
        
    with open(json_path, 'w') as f:
        json.dump(out, f)


def _load_tracking_results(result_txt_path):
    tracks_by_frame = defaultdict(lambda: {"tlwhs": [], "track_ids": [], "scores": []})
    with open(result_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            fields = line.split(",")
            if len(fields) < 7: continue
            frame_id = int(float(fields[0]))
            track_id = int(float(fields[1]))
            tracks_by_frame[frame_id]["tlwhs"].append([float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])])
            tracks_by_frame[frame_id]["track_ids"].append(track_id)
            tracks_by_frame[frame_id]["scores"].append(float(fields[6]))
    return tracks_by_frame


def _load_branch_results(branch_csv_path):
    branch_by_frame = {}
    if not os.path.exists(branch_csv_path): return branch_by_frame
    with open(branch_csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("frame,"): continue
            frame_str, branch_label = line.split(",", 1)
            branch_by_frame[int(float(frame_str))] = branch_label.strip()
    return branch_by_frame


def _build_sequence_frames_from_annotations(data_dir, split_name, ann_file):
    ann_path = os.path.join(data_dir, "annotations", ann_file)
    with open(ann_path, "r") as f: ann = json.load(f)
    seq_frames = defaultdict(list)
    for image_info in ann.get("images", []):
        file_name = image_info.get("file_name", "")
        if not file_name: continue
        seq_frames[file_name.split("/")[0]].append((int(image_info.get("frame_id", 0)), os.path.join(data_dir, split_name, file_name)))
    for seq_name in seq_frames: seq_frames[seq_name].sort(key=lambda x: x[0])
    return seq_frames


def get_target_count(frame_tracks: dict) -> int:
    if frame_tracks is None:
        return 0
    return len(frame_tracks.get("tlwhs", []))


def append_branch_telemetry(vis_frame: np.ndarray, frame_id: int, pipeline_fps: float, target_count: int, branch_label: str) -> None:
    base_string = f"frame: {frame_id} fps: {pipeline_fps:.2f} num: {target_count}"
    offset_x = cv2.getTextSize(base_string, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0][0] + 6
    cv2.putText(vis_frame, f", {branch_label}", (offset_x, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


def render_tracking_visualizations(exp, results_folder, vis_output_dir, pipeline_fps, make_video=False, video_fps=30):
    seq_frames = _build_sequence_frames_from_annotations(exp.data_dir, "test", exp.val_ann)
    os.makedirs(vis_output_dir, exist_ok=True)
    result_files = sorted(glob.glob(os.path.join(results_folder, "*.txt")))
    
    total_images = 0
    for result_txt in result_files:
        seq_name = os.path.splitext(os.path.basename(result_txt))[0]
        if seq_name not in seq_frames: 
            continue

        tracks_by_frame = _load_tracking_results(result_txt)
        branch_by_frame = _load_branch_results(os.path.join(results_folder, f"{seq_name}_branch.csv"))
        sequence_output_dir = os.path.join(vis_output_dir, seq_name)
        os.makedirs(sequence_output_dir, exist_ok=True)

        video_writer, video_path = None, os.path.join(vis_output_dir, f"{seq_name}.mp4")

        for frame_id, image_path in seq_frames[seq_name]:
            frame = cv2.imread(image_path)
            if frame is None: 
                continue

            frame_tracks = tracks_by_frame.get(frame_id)
            branch_label = branch_by_frame.get(frame_id, "Full")
            
            vis_frame = plot_tracking(
                frame, 
                [] if frame_tracks is None else frame_tracks["tlwhs"], 
                [] if frame_tracks is None else frame_tracks["track_ids"], 
                scores=None if frame_tracks is None else frame_tracks["scores"], 
                frame_id=frame_id, 
                fps=pipeline_fps
            )

            target_count = get_target_count(frame_tracks)
            append_branch_telemetry(vis_frame, frame_id, pipeline_fps, target_count, branch_label)

            cv2.imwrite(os.path.join(sequence_output_dir, os.path.basename(image_path)), vis_frame)
            total_images += 1

            if make_video:
                if video_writer is None:
                    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
                    if fourcc_fn is not None: 
                        video_writer = cv2.VideoWriter(video_path, fourcc_fn(*"mp4v"), float(video_fps), (vis_frame.shape[1], vis_frame.shape[0]))
                if video_writer is not None: 
                    video_writer.write(vis_frame)

        if video_writer is not None: 
            video_writer.release()


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if not hasattr(args, 'distance') or args.distance == 'nwd':
        if hasattr(exp, 'distance'): args.distance = exp.distance
    mode_name = "early_exit" if args.early_exit else "baseline"
    args.experiment_name = f"{args.experiment_name}_{mode_name}_{args.distance}"
    is_distributed = num_gpu > 1
    cudnn.benchmark = True
    rank = args.local_rank

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    if rank == 0: os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None: exp.test_conf = args.conf
    if args.nms is not None: exp.nmsthre = args.nms
    if args.tsize is not None: exp.test_size = (args.tsize, args.tsize)
    exp.early_exit_enabled = args.early_exit

    model = exp.get_model()
    model_info = get_model_info(model, exp.test_size)
    logger.info("Model Summary: {}".format(model_info))
    full_network_gflops = getattr(exp, "full_network_gflops", parse_gflops_from_model_info(model_info))
    p3_only_gflops = getattr(exp, "p3_only_gflops", None)

    ensure_coco_json_exists(exp.data_dir, exp.val_ann, args.mot20)
    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)
    
    # MOTEvaluator from aerotrack
    evaluator = MOTEvaluator(
        args=args, dataloader=val_loader, img_size=exp.test_size,
        confthre=exp.test_conf, nmsthre=exp.nmsthre, num_classes=exp.num_classes
    )

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar") if args.ckpt is None else args.ckpt
        logger.info("loading checkpoint")
        model.load_state_dict(torch.load(ckpt_file, map_location=f"cuda:{rank}")["model_state_dict"], strict=False)
        logger.info("loaded checkpoint done.")

    if is_distributed: model = DDP(model, device_ids=[rank])
    if args.fuse: model = fuse_model(model)

    if args.trt:
        trt_file = os.path.join(file_name, "model_trt.pth")
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file, decoder = None, None

    logger.info("⏱️ Starting performance timer...")
    t0 = time()
    *_, summary_coco = evaluator.evaluate(model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder)
    t1 = time()
    
    total_time = t1 - t0
    num_frames = len(val_loader.dataset)
    pipeline_latency = (total_time / max(num_frames, 1)) * 1000.0
    pipeline_fps = num_frames / max(total_time, 1e-9)
    
    hardware_latency = None
    match = re.search(r"Average inference time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", summary_coco)
    if match:
        hardware_latency = float(match.group(1))

    logger.info("\n" + summary_coco)

    if hasattr(evaluator, "last_early_stats"):
        stats = evaluator.last_early_stats
        total_early, total_total = 0, 0
        early_stats_csv = os.path.join(file_name, "early_exit_stats.csv")
        
        csv_lines = ["sequence,early_exits,total_frames,early_exit_ratio\n"]
        for vname, s in stats.items():
            if s["total"] > 0:
                ratio = s["early"] / s["total"]
                total_early += s["early"]
                total_total += s["total"]
                csv_lines.append(f"{vname},{s['early']},{s['total']},{ratio:.6f}\n")
                logger.info(f"early exits {s['early']}/{s['total']} ({ratio:.1%}) for {vname}")
                
        with open(early_stats_csv, "w") as f:
            f.writelines(csv_lines)
            
        if total_total > 0:
            effective_gflops, gflops_reduction, exit_rate = compute_effective_gflops(total_early, total_total, full_network_gflops, p3_only_gflops)
            if effective_gflops is not None and gflops_reduction is not None:
                logger.info(f"early exits {total_early}/{total_total} ({exit_rate:.1%}) | Effective GFLOPs: {effective_gflops:.2f} | GFLOPs reduction: {gflops_reduction:.2f}%")
            else:
                logger.info(f"early exits {total_early}/{total_total} ({total_early/total_total:.1%})")

    frame_latency_summary = summarize_frame_latency_records(getattr(evaluator, "last_frame_latency_records", []))
    if frame_latency_summary is not None:
        best_frame = frame_latency_summary["best"]
        worst_frame = frame_latency_summary["worst"]
        logger.info(
            "Frame latency extremes | best: {} frame {} ({:.2f} ms, {:.2f} FPS) | worst: {} frame {} ({:.2f} ms, {:.2f} FPS)".format(
                best_frame["sequence"], best_frame["frame_id"], best_frame["latency_ms"], best_frame["fps"],
                worst_frame["sequence"], worst_frame["frame_id"], worst_frame["latency_ms"], worst_frame["fps"],
            )
        )

    mm.lap.default_solver = 'lap'
    gt_type = '_val_half' if exp.val_ann == 'val_half.json' else ''
    
    split_folder = 'test' 
    gt_pattern = os.path.join(exp.data_dir, split_folder, f'*/gt/gt{gt_type}.txt')
    gtfiles = glob.glob(gt_pattern)
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

    logger.info(f'Recherche des fichiers GT avec le chemin : {gt_pattern}')
    
    gt = OrderedDict([(Path(f).parent.parent.name, mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).name)[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])    
    
    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logger.info('Running metrics')
    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary_df = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']
    }
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary_df[divided] = (summary_df[divided] / summary_df[divisor])
            
    if hardware_latency is not None:
        summary_df.loc['OVERALL', 'HW_Latency_ms'] = round(hardware_latency, 2)
        summary_df.loc['OVERALL', 'HW_FPS'] = round(1000.0 / hardware_latency, 2)

    if frame_latency_summary is not None:
        summary_df.loc['OVERALL', 'Best_Frame_Sequence'] = best_frame["sequence"]
        summary_df.loc['OVERALL', 'Best_Frame_Id'] = int(best_frame["frame_id"])
        summary_df.loc['OVERALL', 'Best_Frame_Latency_ms'] = round(best_frame["latency_ms"], 2)
        summary_df.loc['OVERALL', 'Best_Frame_FPS'] = round(best_frame["fps"], 2)
        summary_df.loc['OVERALL', 'Worst_Frame_Sequence'] = worst_frame["sequence"]
        summary_df.loc['OVERALL', 'Worst_Frame_Id'] = int(worst_frame["frame_id"])
        summary_df.loc['OVERALL', 'Worst_Frame_Latency_ms'] = round(worst_frame["latency_ms"], 2)
        summary_df.loc['OVERALL', 'Worst_Frame_FPS'] = round(worst_frame["fps"], 2)
        
    summary_df.loc['OVERALL', 'Pipeline_Latency_ms'] = round(pipeline_latency, 2)
    
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked', 'partially_tracked', 'mostly_lost']
    for k in change_fmt_list: fmt[k] = fmt['mota']
        
    print(mm.io.render_summary(summary_df, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    csv_output_path = os.path.join(file_name, "mot_evaluation_metrics.csv")
    summary_df.to_csv(csv_output_path)

    if rank == 0 and args.save_vis:
       render_tracking_visualizations(
           exp, 
           results_folder, 
           args.vis_output if args.vis_output else os.path.join(file_name, "track_vis"), 
           pipeline_fps=pipeline_fps,
           make_video=args.vis_video, 
           video_fps=args.vis_fps
       )
    logger.info(f"Latency: {hardware_latency:.2f} ms/img | FPS: {1000.0/hardware_latency:.2f}")
    logger.info('Completed')


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    if not args.experiment_name: args.experiment_name = exp.exp_name
    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    launch(main, num_gpu, args.num_machines, args.machine_rank, backend=args.dist_backend, dist_url=args.dist_url, args=(exp, args, num_gpu))