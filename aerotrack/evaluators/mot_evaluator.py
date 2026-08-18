from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from aerotrack.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from aerotrack.tracker.byte_tracker import BYTETracker


import contextlib
import io
import os
import itertools
import json
import tempfile
import time


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    logger.info('saving results to {}'.format(os.path.abspath(filename)))
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    logger.info('saving results to {}'.format(os.path.abspath(filename)))
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def summarize_frame_latency_records(frame_latency_records):
    if not frame_latency_records:
        return None

    best_record = min(frame_latency_records, key=lambda record: record['latency_ms'])
    worst_record = max(frame_latency_records, key=lambda record: record['latency_ms'])

    return {
        'count': len(frame_latency_records),
        'best': best_record,
        'worst': worst_record,
    }


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def estimate_flops_reduction(self, early_count, total_count):
        """
        Estimate FLOPS reduction from early exits.
        Assumes early exit skips ~60% of computation (rough estimate for YOLOX architecture).
        """
        if total_count == 0:
            return 0.0
        early_ratio = early_count / total_count
        # Estimate: full path = 100%, early path = 40% (skips P4, P5, PAFPN, some head ops)
        flops_reduction = early_ratio * 0.60
        return flops_reduction * 100.0

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        from collections import defaultdict

        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
            
        data_list = []
        # Utilisation d'un dictionnaire pour s'assurer que chaque vidéo a sa liste (même vide)
        results_by_video = defaultdict(list)
        branch_by_video = defaultdict(list)
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        frame_latency_records = []
        warmup_frames = 20
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            import tensorrt as trt
            logger.info(f"Chargement du moteur TensorRT natif depuis : {trt_file}")
            with open(trt_file, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
                context = engine.create_execution_context()
            model = context
            
        distance_metric = getattr(self.args, 'distance', 'nwd')
        tracker = BYTETracker(self.args, distance_metric=distance_metric)
        ori_thresh = self.args.track_thresh
        prev_video_name = None
        # track early-exit statistics per video
        early_stats = defaultdict(lambda: {"early": 0, "total": 0})
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                # 1. Extraction du nom de la vidéo et du numéro de frame
                file_path = info_imgs[4][0]
                video_name = file_path.split('/')[0] 
                frame_str = file_path.split('/')[-1].split('.')[0]
                frame_id = int(frame_str) 

                # 2. Enregistrer la vidéo dans le dictionnaire (même s'il n'y a pas de détection)
                if video_name not in results_by_video:
                    results_by_video[video_name] = []

                # 3. Réinitialisation du tracker au changement de vidéo
                if video_name != prev_video_name:
                    distance_metric = getattr(self.args, 'distance', 'nwd')
                    tracker = BYTETracker(self.args, distance_metric=distance_metric)
                    prev_video_name = video_name

                # Paramètres génériques
                self.args.track_buffer = 30
                self.args.track_thresh = ori_thresh

                imgs = imgs.type(tensor_type)

                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                # --- Inférence du modèle ---
                outputs = model(imgs)
                # capture whether model used early exit for this forward (may be None for TRT)
                early_exit = getattr(model, "last_early_exit", None)
                if early_exit is not None:
                    branch_by_video[video_name].append((frame_id, "EE" if early_exit else "Full"))
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # --- Suivi (Tracking) UNIQUEMENT SI détection ---
            if outputs[0] is not None:
                if is_time_record:
                    track_start = time_synchronized()
                
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
                
                if is_time_record:
                    track_end = time_synchronized()
                    track_time += track_end - track_start

                if is_time_record and frame_id > warmup_frames:
                    frame_end = track_end if outputs[0] is not None else infer_end
                    frame_latency_ms = (frame_end - start) * 1000.0
                    frame_latency_records.append({
                        'sequence': video_name,
                        'frame_id': frame_id,
                        'latency_ms': frame_latency_ms,
                        'fps': 1000.0 / max(frame_latency_ms, 1e-9),
                    })
                    
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    if tlwh[2] * tlwh[3] > self.args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                
                # On ajoute les prédictions dans le dictionnaire
                results_by_video[video_name].append((frame_id, online_tlwhs, online_ids, online_scores))

                # update early-exit stats for this video (count one forward)
                if early_exit is not None:
                    early_stats[video_name]["total"] += 1
                    if early_exit:
                        early_stats[video_name]["early"] += 1

        # 4. ÉCRITURE FORCÉE DE TOUS LES FICHIERS
        # On boucle sur toutes les vidéos vues, même celles avec une liste vide
        for v_name in results_by_video.keys():
            res_file = os.path.join(result_folder, f"{v_name}.txt")
            write_results(res_file, results_by_video[v_name])

        for v_name in branch_by_video.keys():
            branch_file = os.path.join(result_folder, f"{v_name}_branch.csv")
            logger.info('saving branch labels to {}'.format(os.path.abspath(branch_file)))
            with open(branch_file, 'w') as f:
                f.write('frame,branch\n')
                for frame_id, branch_label in branch_by_video[v_name]:
                    f.write(f'{frame_id},{branch_label}\n')



        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            frame_latency_records = gather(frame_latency_records, dst=0)
            frame_latency_records = list(itertools.chain(*frame_latency_records))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        self.last_frame_latency_records = frame_latency_records
        self.last_frame_latency_summary = summarize_frame_latency_records(frame_latency_records)
        # store early-exit stats and calculate FLOPS reduction
        self.last_early_stats = early_stats
        total_early = sum(s["early"] for s in early_stats.values())
        total_total = sum(s["total"] for s in early_stats.values())
        if total_total > 0:
            flops_reduction = self.estimate_flops_reduction(total_early, total_total)
            self.last_flops_reduction = flops_reduction
        else:
            self.last_flops_reduction = 0.0
        synchronize()
        return eval_results
    
    def evaluate_sort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            import tensorrt as trt
            logger.info(f"Chargement du moteur TensorRT natif depuis : {trt_file}")
            with open(trt_file, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
                context = engine.create_execution_context()
            model = context
            
        tracker = Sort(self.args.track_thresh)
        prev_video_id = None
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name

                if video_id != prev_video_id:
                    distance_metric = getattr(self.args, 'distance', 'nwd')
                    tracker = BYTETracker(self.args, distance_metric=distance_metric)
                    if prev_video_id is not None and len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[prev_video_id]))
                        write_results(result_filename, results)
                    results = []  # On vide la liste pour la nouvelle vidéo
                    prev_video_id = video_id

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                last_vid = prev_video_id if prev_video_id is not None else video_id
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[last_vid]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_deepsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            import tensorrt as trt
            logger.info(f"Chargement du moteur TensorRT natif depuis : {trt_file}")
            with open(trt_file, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
                context = engine.create_execution_context()
            model = context
            
        tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
        prev_video_id = None
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name

                if frame_id == 1:
                    tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
                    if prev_video_id is not None and len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[prev_video_id]))
                        write_results_no_score(result_filename, results)
                        results = []
                    prev_video_id = video_id

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                last_vid = prev_video_id if prev_video_id is not None else video_id
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[last_vid]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_motdt(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            import tensorrt as trt
            logger.info(f"Chargement du moteur TensorRT natif depuis : {trt_file}")
            with open(trt_file, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
                context = engine.create_execution_context()
            model = context
            
        tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
        prev_video_id = None
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name

                if frame_id == 1:
                    tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
                    if prev_video_id is not None and len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[prev_video_id]))
                        write_results(result_filename, results)
                        results = []
                    prev_video_id = video_id

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                last_vid = prev_video_id if prev_video_id is not None else video_id
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[last_vid]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        cocoGt = self.dataloader.dataset.coco
        ann_ids = cocoGt.getAnnIds()
        logger.info(f"DEBUG: Nombre total d'annotations trouvées dans le JSON : {len(ann_ids)}")
            
        if len(ann_ids) == 0:
            logger.error("ALERTE : Aucune annotation n'a été chargée depuis ton fichier JSON !")

        logger.info("Evaluate in main process...")
        logger.info(f"Nombre total de détections collectées : {len(data_dict)}")
        if len(data_dict) == 0:
            logger.warning("Attention : aucune détection n'a été transmise à l'évaluateur !")
        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            '''
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            #from pycocotools.cocoeval import COCOeval
            from aerotrack.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info

            