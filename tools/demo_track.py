import argparse
import os
import time
import re
import cv2
import torch
from loguru import logger

from aerotrack.data.data_augment import preproc
from aerotrack.exp import get_exp
from aerotrack.utils import fuse_model, get_model_info, postprocess
from aerotrack.utils.energy import EnergyMonitor
from aerotrack.evaluators.mot_evaluator import summarize_frame_latency_records
from aerotrack.utils.visualize import plot_tracking
from aerotrack.tracker.byte_tracker import BYTETracker
from aerotrack.tracking_utils.timer import Timer

def build_demo_parser():
    parser = argparse.ArgumentParser("AeroTrack Live Demonstration Engine")
    parser.add_argument("demo", default="video", choices=["image", "video", "webcam"])
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("--path", default="./videos/uav_swarm_input.mp4")
    parser.add_argument("--camid", type=int, default=0)
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("-f", "--exp_file", type=str, required=True)
    parser.add_argument("-c", "--ckpt", type=str, required=True)
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--nms", type=float, default=None)
    parser.add_argument("--tsize", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--trt", action="store_true")
    parser.add_argument("--track_thresh", type=float, default=0.5)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6)
    parser.add_argument("--min_box_area", type=float, default=10)
    parser.add_argument("--mot20", action="store_true")
    
    # --- AJOUTS : Options pour l'architecture AeroTrack ---
    parser.add_argument("--distance", type=str, default="nwd", choices=["nwd", "iou"], help="distance metric for tracking")
    parser.add_argument("--early_exit", dest="early_exit", default=False, action="store_true", help="Activer le routage dynamique Early Exit.")
    return parser

def extract_base_gflops(model_info_string):
    match = re.search(r"Gflops:\s*([0-9]+(?:\.[0-9]+)?)", model_info_string)
    if match is None:
        return None
    return float(match.group(1))

def calculate_effective_computation(early_exit_count, total_frames, base_gflops, shallow_gflops_ratio=0.40):
    if total_frames <= 0 or base_gflops is None:
        return None, None, None
    early_exit_rate = early_exit_count / total_frames
    deep_routing_rate = 1.0 - early_exit_rate
    shallow_gflops = base_gflops * shallow_gflops_ratio
    effective_gflops = (early_exit_rate * shallow_gflops) + (deep_routing_rate * base_gflops)
    gflops_reduction_percentage = (1.0 - (effective_gflops / base_gflops)) * 100.0
    return effective_gflops, gflops_reduction_percentage, early_exit_rate

class VisualInferenceEngine:
    def __init__(self, detection_model, experiment_configuration, execution_device, half_precision_enabled):
        self.detection_model = detection_model
        self.num_classes = experiment_configuration.num_classes
        self.confidence_threshold = experiment_configuration.test_conf
        self.nms_threshold = experiment_configuration.nmsthre
        self.target_input_dimensions = experiment_configuration.test_size
        self.execution_device = execution_device
        self.half_precision_enabled = half_precision_enabled

    def process_frame(self, raw_frame, performance_timer):
        frame_metadata = {
            "raw_height": raw_frame.shape[0],
            "raw_width": raw_frame.shape[1],
            "raw_frame": raw_frame
        }
        
        normalized_tensor, processing_scale_ratio = preproc(raw_frame, self.target_input_dimensions)
        frame_metadata["ratio"] = processing_scale_ratio
        
        execution_tensor = torch.from_numpy(normalized_tensor).unsqueeze(0).float().to(self.execution_device)
        
        if self.half_precision_enabled:
            execution_tensor = execution_tensor.half()
            
        with torch.no_grad():
            performance_timer.tic()
            raw_predictions = self.detection_model(execution_tensor)
            
            # Safely extract early exit telemetry without crashing the baseline
            frame_metadata["early_exit_triggered"] = getattr(self.detection_model, "last_early_exit", None)
            
            processed_detections = postprocess(raw_predictions, self.num_classes, self.confidence_threshold, self.nms_threshold)
            
        return processed_detections, frame_metadata

def execute_video_stream(inference_engine, output_directory, execution_arguments, experiment_configuration, base_gflops):
    video_capture = cv2.VideoCapture(execution_arguments.path if execution_arguments.demo == "video" else execution_arguments.camid)
    video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_folder = os.path.join(output_directory, timestamp)
    os.makedirs(save_folder, exist_ok=True)

    video_writer = None
    if execution_arguments.save_result:
        save_path = os.path.join(
            save_folder,
            os.path.basename(execution_arguments.path) if execution_arguments.demo == "video" else "camera.mp4",
        )
        fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
        if fourcc_fn is not None:
            video_writer = cv2.VideoWriter(save_path, fourcc_fn(*"mp4v"), video_fps, (int(video_width), int(video_height)))

    tracking_engine = BYTETracker(execution_arguments, frame_rate=30, distance_metric=execution_arguments.distance)
    performance_timer = Timer()

    current_frame_index = 0
    tracking_results = []
    frame_latency_records = []
    early_exit_count = 0
    warmup_frames = 20
    energy_monitor = EnergyMonitor(sample_interval_s=0.5)
    energy_monitor.start()
    energy_summary = None

    try:
        while True:
            frame_retrieved, raw_frame = video_capture.read()
            if not frame_retrieved:
                break

            frame_start = time.perf_counter()
            detections, metadata = inference_engine.process_frame(raw_frame, performance_timer)

            early_exit_triggered = metadata.get("early_exit_triggered")
            if early_exit_triggered is True:
                early_exit_count += 1

            if detections[0] is not None:
                active_tracks = tracking_engine.update(
                    detections[0],
                    [metadata["raw_height"], metadata["raw_width"]],
                    inference_engine.target_input_dimensions,
                )
                valid_tlwhs = []
                valid_ids = []

                for track in active_tracks:
                    vertical_ratio_violation = (track.tlwh[2] / track.tlwh[3]) > execution_arguments.aspect_ratio_thresh
                    if (track.tlwh[2] * track.tlwh[3] > execution_arguments.min_box_area) and not vertical_ratio_violation:
                        valid_tlwhs.append(track.tlwh)
                        valid_ids.append(track.track_id)
                        tracking_results.append(
                            f"{current_frame_index},{track.track_id},{track.tlwh[0]:.2f},{track.tlwh[1]:.2f},{track.tlwh[2]:.2f},{track.tlwh[3]:.2f},{track.score:.2f},-1,-1,-1\n"
                        )

                performance_timer.toc()
                frame_end = time.perf_counter()
                frame_latency_ms = (frame_end - frame_start) * 1000.0
                if current_frame_index + 1 > warmup_frames:
                    frame_latency_records.append(
                        {
                            "sequence": os.path.basename(execution_arguments.path) if execution_arguments.demo == "video" else "camera",
                            "frame_id": current_frame_index + 1,
                            "latency_ms": frame_latency_ms,
                            "fps": 1000.0 / max(frame_latency_ms, 1e-9),
                        }
                    )
                annotated_frame = plot_tracking(
                    metadata["raw_frame"],
                    valid_tlwhs,
                    valid_ids,
                    frame_id=current_frame_index + 1,
                    fps=1.0 / max(1e-5, performance_timer.average_time),
                )
            else:
                performance_timer.toc()
                frame_end = time.perf_counter()
                frame_latency_ms = (frame_end - frame_start) * 1000.0
                if current_frame_index + 1 > warmup_frames:
                    frame_latency_records.append(
                        {
                            "sequence": os.path.basename(execution_arguments.path) if execution_arguments.demo == "video" else "camera",
                            "frame_id": current_frame_index + 1,
                            "latency_ms": frame_latency_ms,
                            "fps": 1000.0 / max(frame_latency_ms, 1e-9),
                        }
                    )
                annotated_frame = metadata["raw_frame"]

            if early_exit_triggered is not None:
                status_label = "EE" if early_exit_triggered else "Full"
                status_color = (0, 165, 255) if early_exit_triggered else (0, 255, 0)
                cv2.putText(annotated_frame, f"[{status_label}]", (10, 75), cv2.FONT_HERSHEY_PLAIN, 2, status_color, thickness=2)

            if execution_arguments.save_result and video_writer is not None:
                video_writer.write(annotated_frame)

            cv2.imshow("AeroTrack Real-Time Multi-Target Stream", annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            current_frame_index += 1
    finally:
        energy_summary = energy_monitor.stop()
        video_capture.release()
        if video_writer is not None:
            video_writer.release()

    if execution_arguments.save_result:
        results_file_path = os.path.join(output_directory, f"{timestamp}.txt")
        with open(results_file_path, "w") as output_file:
            output_file.writelines(tracking_results)

    if current_frame_index > 0 and early_exit_count > 0:
        effective_gflops, gflops_reduction, exit_rate = calculate_effective_computation(
            early_exit_count,
            current_frame_index,
            base_gflops,
        )
        if effective_gflops is not None:
            logger.info(
                f"Early Exits: {early_exit_count}/{current_frame_index} ({exit_rate:.1%}) | Effective GFLOPs: {effective_gflops:.2f} | GFLOPs Reduction: {gflops_reduction:.2f}%"
            )

    frame_latency_summary = summarize_frame_latency_records(frame_latency_records)
    best_frame = None
    worst_frame = None
    if frame_latency_summary is not None:
        best_frame = frame_latency_summary["best"]
        worst_frame = frame_latency_summary["worst"]
        logger.info(
            "Frame latency extremes | best: {} frame {} ({:.2f} ms, {:.2f} FPS) | worst: {} frame {} ({:.2f} ms, {:.2f} FPS)".format(
                best_frame["sequence"],
                best_frame["frame_id"],
                best_frame["latency_ms"],
                best_frame["fps"],
                worst_frame["sequence"],
                worst_frame["frame_id"],
                worst_frame["latency_ms"],
                worst_frame["fps"],
            )
        )

    if energy_summary is not None:
        logger.info(
            "Energy usage | backend: {} | energy: {:.2f} J | avg power: {:.2f} W | peak power: {:.2f} W | samples: {}".format(
                energy_summary.backend,
                energy_summary.energy_j,
                energy_summary.average_power_w,
                energy_summary.peak_power_w,
                energy_summary.sample_count,
            )
        )

    performance_summary_path = os.path.join(output_directory, f"{timestamp}_performance_summary.csv")
    with open(performance_summary_path, "w") as summary_file:
        summary_file.write("metric,sequence,frame_id,latency_ms,fps,energy_j,avg_power_w,peak_power_w,backend\n")
        if best_frame is not None and worst_frame is not None:
            summary_file.write(
                f"best,{best_frame['sequence']},{best_frame['frame_id']},{best_frame['latency_ms']:.2f},{best_frame['fps']:.2f},,,,,\n"
            )
            summary_file.write(
                f"worst,{worst_frame['sequence']},{worst_frame['frame_id']},{worst_frame['latency_ms']:.2f},{worst_frame['fps']:.2f},,,,,\n"
            )
        if energy_summary is not None:
            summary_file.write(
                f"energy,,,,,{energy_summary.energy_j:.2f},{energy_summary.average_power_w:.2f},{energy_summary.peak_power_w:.2f},{energy_summary.backend}\n"
            )

def main():
    parsed_arguments = build_demo_parser().parse_args()
    active_experiment = get_exp(parsed_arguments.exp_file, parsed_arguments.name)
    
    # --- AJOUT : Nommer le dossier de sortie comme dans track.py ---
    if not hasattr(parsed_arguments, 'distance') or parsed_arguments.distance == 'nwd':
        if hasattr(active_experiment, 'distance'): parsed_arguments.distance = active_experiment.distance
        
    mode_name = "early_exit" if parsed_arguments.early_exit else "baseline"
    
    if not parsed_arguments.experiment_name:
        parsed_arguments.experiment_name = active_experiment.exp_name
        
    parsed_arguments.experiment_name = f"{parsed_arguments.experiment_name}_{mode_name}_{parsed_arguments.distance}"
    # ---------------------------------------------------------------
    
    target_output_directory = os.path.join(active_experiment.output_dir, parsed_arguments.experiment_name)
    os.makedirs(target_output_directory, exist_ok=True)
    
    visualization_folder = os.path.join(target_output_directory, "track_vis") if parsed_arguments.save_result else None
    if visualization_folder:
        os.makedirs(visualization_folder, exist_ok=True)
        
    computed_device = torch.device("cuda" if parsed_arguments.device == "gpu" else "cpu")
    
    if parsed_arguments.conf is not None:
        active_experiment.test_conf = parsed_arguments.conf
    if parsed_arguments.nms is not None:
        active_experiment.nmsthre = parsed_arguments.nms
    if parsed_arguments.tsize is not None:
        active_experiment.test_size = (parsed_arguments.tsize, parsed_arguments.tsize)
        
    # --- AJOUT : Activer dynamiquement le Early Exit avant la création du modèle ---
    active_experiment.early_exit_enabled = parsed_arguments.early_exit
    
    configured_model = active_experiment.get_model().to(computed_device)
    model_architecture_info = get_model_info(configured_model, active_experiment.test_size)
    base_gflops = extract_base_gflops(model_architecture_info)
    
    configured_model.eval()
    
    if not parsed_arguments.trt:
        checkpoint_payload = torch.load(parsed_arguments.ckpt, map_location="cpu")
        configured_model.load_state_dict(checkpoint_payload.get("model_state_dict", checkpoint_payload.get("model")))
        
    if parsed_arguments.fuse:
        configured_model = fuse_model(configured_model)
    if parsed_arguments.fp16:
        configured_model = configured_model.half()
        
    inference_engine = VisualInferenceEngine(configured_model, active_experiment, computed_device, parsed_arguments.fp16)
    
    if parsed_arguments.demo in ["video", "webcam"]:
        execute_video_stream(inference_engine, visualization_folder if visualization_folder else target_output_directory, parsed_arguments, active_experiment, base_gflops)

if __name__ == "__main__":
    main()