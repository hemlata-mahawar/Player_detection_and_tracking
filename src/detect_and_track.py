import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import torch
import yaml
from pathlib import Path
import json
from tqdm import tqdm
import time
from src.mask_select import OvalROISelector
from pathlib import Path


class CricketPlayerTracker:
    def __init__(self, model_config='configs/yolov12_config.yaml', tracker_config='configs/botsort.yaml'):
        """
        Initialize the cricket player tracker with config files.
        
        Args:
            model_config: Path to YOLO model configuration
            tracker_config: Path to tracker configuration file
        """
        print("Loading configuration...")
        
        # Load model configuration
        with open(model_config, 'r') as f:
            self.model_config = yaml.safe_load(f)
        
        # Load tracker configuration
        with open(tracker_config, 'r') as f:
            self.tracker_config = yaml.safe_load(f)
        
        print(f"Loading YOLO model: {self.model_config.get('model')}")
        
        # Initialize model with config
        self.model = YOLO(self.model_config.get('model'))
        
        # Initialize tracking history
        self.tracks_history = {}
        
    def detect_and_track(self, video_path, output_path=None, show_video=False, save_output=True, show_mask=True):
        """
        Process video to detect and track cricket players.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            show_video: Whether to display video during processing
            save_output: Whether to save output video
        
        Returns:
            Dictionary containing tracking data
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
        # Prepare output video writer
        if save_output:
            if output_path is None:
                output_path = Path(video_path).stem + "_tracked.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read the first frame of the video.")
        
        mask_dir = Path('data')
        MASK_PATH = mask_dir / f"{Path(video_path).stem}_pitch_mask.npy"
        if Path(MASK_PATH).exists():
            print("Loading existing pitch mask...")
            self.pitch_mask = np.load(MASK_PATH)

        else:
            print("No saved mask found. Creating new mask...")
            roi_selector = OvalROISelector(first_frame)
            self.pitch_mask = roi_selector.get_mask()

            if self.pitch_mask is None:
                raise ValueError("No pitch mask was created.")

            np.save(MASK_PATH, self.pitch_mask)
            print(f"Pitch mask saved to {MASK_PATH}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
        
        frame_count = 0
        tracking_data = {}
        processing_times = []
        # Create progress bar
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()            
            # Perform tracking with config parameters
            results = self.model.track(
                source=frame,
                persist=True,
                # tracker='botsort.yaml',
                tracker=self.tracker_config.get('tracker', 'configs/botsort.yaml'),
                classes=self.model_config.get('classes', [0]),
                conf=self.model_config.get('conf', 0.15),
                iou=self.model_config.get('iou', 0.5),
                imgsz=self.model_config.get('imgsz'),
                show_labels=self.model_config.get('show_labels', False),
                verbose=False,
                device=self.model_config.get('device', 'gpu'),  # Change to 'cuda' if GPU available
                max_det=self.model_config.get('max_det', 30),
                agnostic_nms=self.model_config.get('agnostic_nms', False),
                half = self.model_config.get('half', False),
                dnn = self.model_config.get('dnn', False),
                show_conf=self.model_config.get('show_conf', False),
                show_boxes=self.model_config.get('show_boxes', True),
                line_width=self.model_config.get('line_width', 2),
                # save = self.model_config.get('save', False),
                save_conf = self.model_config.get('save_conf', False),
                save_crop = self.model_config.get('save_crop', False),
                format = self.model_config.get('format', 'torchscript'),
                save_txt = self.model_config.get('save_txt', False)
            )
            # print(self.model.predictor.args.tracker)
            # print("Loaded BOT-SORT args:", self.model.predictor.trackers[0].args)

            
            # Calculate processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
                        
            # annotated_frame = frame.copy()

            # Extract tracking information
            if results and len(results) > 0:
                result = results[0]
                
                # Check if tracking data is available
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    # Get bounding boxes
                    if boxes.xyxy is not None:
                        boxes_xyxy = boxes.xyxy.cpu().numpy()
                        
                        # Get track IDs (if available)
                        if hasattr(boxes, 'id') and boxes.id is not None:
                            track_ids = boxes.id.cpu().numpy()
                        else:
                            # Assign temporary IDs if tracking not available
                            track_ids = np.arange(len(boxes_xyxy))
                        
                        # Get confidence scores
                        if hasattr(boxes, 'conf') and boxes.conf is not None:
                            confidences = boxes.conf.cpu().numpy()
                        else:
                            confidences = np.ones(len(boxes_xyxy))

                        filtered_indices = []
                        for i, box in enumerate(boxes_xyxy):
                            x1, y1, x2, y2 = box
                            h, w = self.pitch_mask.shape

                            foot_x1 = int(x1)
                            foot_x2 = int(x2)
                            foot_y = int(y2)

                            # Clamp to valid range
                            foot_x1 = np.clip(foot_x1, 0, w - 1)
                            foot_x2 = np.clip(foot_x2, 0, w - 1)
                            foot_y  = np.clip(foot_y,  0, h - 1)
                            if self.pitch_mask[foot_y, foot_x1] != 0 and self.pitch_mask[foot_y, foot_x2] != 0:
                                filtered_indices.append(i)
                        
                        if len(filtered_indices) > 0:                            
                            result.boxes = result.boxes[filtered_indices]
                            result.names = {0: ""}
                            annotated_frame = result.plot(conf=False)               
                        
                            # Store tracking data
                            tracking_data[frame_count] = []
                            for box, track_id, conf in zip(boxes_xyxy, track_ids, confidences):
                                x1, y1, x2, y2 = box

                                tracking_data[frame_count].append({
                                    'id': int(track_id),
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'confidence': float(conf),
                                    'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)]
                                })

                                if track_id not in self.tracks_history:
                                    self.tracks_history[track_id] = []

                                self.tracks_history[track_id].append({
                                    'frame': frame_count,
                                    'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'confidence': float(conf)
                                })

            if show_mask:
            # Add overlay information
                annotated_frame = self.overlay_mask(
                    annotated_frame,
                    self.pitch_mask,
                    color=(255, 255, 0),
                    alpha=0.1
                )

            annotated_frame = self.add_overlay_info(
                annotated_frame, frame_count, processing_time, 
                len(tracking_data.get(frame_count, []))
            )
            # Show video if requested
            if show_video:
                display_frame = self.prepare_display_frame(annotated_frame, width)
                cv2.imshow('Cricket Player Tracking', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write frame to output video
            if save_output:
                out.write(annotated_frame)
            
            frame_count += 1
            pbar.update(1)
            current_fps = 1/processing_time if processing_time > 0 else 0
            pbar.set_postfix({'FPS': f"{current_fps:.1f}"})
        
        # Release resources
        cap.release()
        if save_output:
            out.release()
        if show_video:
            cv2.destroyAllWindows()
        
        pbar.close()
        
        # Calculate statistics
        total_processing_time = sum(processing_times)
        avg_fps_overall = len(processing_times) / total_processing_time if total_processing_time > 0 else 0
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {avg_fps_overall:.2f}")
        print(f"Unique players tracked: {len(self.tracks_history)}")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        
        return {
            'tracking_data': tracking_data,
            'tracks_history': self.tracks_history,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': frame_count
            },
            'processing_stats': {
                'avg_fps': avg_fps_overall,
                'total_time': total_processing_time,
                'config_used': {
                    'model': self.model_config.get('model'),
                    'imgsz': self.model_config.get('imgsz'),
                    'conf': self.model_config.get('conf'),
                    'iou': self.model_config.get('iou')
                }
            }
        }

    def overlay_mask(self, frame, mask, color=(255, 255, 0), alpha=0.3):
        overlay = frame.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    def add_overlay_info(self, frame, frame_count, processing_time, player_count):
        """Add overlay information to frame."""
        # Add FPS counter
        fps = 1/processing_time if processing_time > 0 else 0
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add player count
        cv2.putText(frame, f"Players: {player_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add config info (top right)
        config_text = f"Conf: {self.model_config.get('conf', 0.15)}, IOU: {self.model_config.get('iou', 0.5)}"
        text_size = cv2.getTextSize(config_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = frame.shape[1] - text_size[0] - 10
        cv2.putText(frame, config_text, (text_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def prepare_display_frame(self, frame, original_width, max_display_width=1280):
        """Prepare frame for display by resizing if too large."""
        if original_width > max_display_width:
            scale = max_display_width / original_width
            new_width = max_display_width
            new_height = int(frame.shape[0] * scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame
    
    def save_tracking_data(self, output_path='tracking_data.json'):
        """
        Save tracking data to JSON file.
        """
        # Convert to serializable format
        serializable_history = {}
        for track_id, history in self.tracks_history.items():
            serializable_history[str(track_id)] = history
        
        data = {
            'tracks_history': serializable_history,
            'num_tracks': len(self.tracks_history),
            'config': {
                'model': self.model_config.get('model'),
                'detection': {
                    'imgsz': self.model_config.get('imgsz'),
                    'conf': self.model_config.get('conf'),
                    'iou': self.model_config.get('iou'),
                    'classes': self.model_config.get('classes', [0])
                },
                'tracking': self.tracker_config
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Tracking data saved to {output_path}")
        
        return data
    
    def update_config(self, param_name, param_value):
        """
        Update configuration parameter dynamically.
        
        Args:
            param_name: Parameter name to update
            param_value: New value
        """
        if param_name in ['conf', 'iou', 'imgsz']:
            setattr(self, param_name, param_value)
            print(f"Updated {param_name} to {param_value}")
        elif param_name == 'classes':
            self.classes = param_value
            print(f"Updated classes to {param_value}")
        else:
            print(f"Parameter {param_name} not recognized")

if __name__ == "__main__":
    # Example usage with config files
    tracker = CricketPlayerTracker(
        model_config='configs/yolov12_config.yaml',
        tracker_config='configs/botsort.yaml'
    )
    
    # Process video
    results = tracker.detect_and_track(
        video_path='data/input_video.mp4',
        output_path='output/tracked_video.mp4',
        show_video=False,
        save_output=True
    )
    
    # Save tracking data
    tracker.save_tracking_data('output/tracking_data.json')