import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

class VisualizationTools:
    @staticmethod
    def create_tracking_video_with_trajectories(video_path, tracking_data, output_path, 
                                               trajectory_length=30):
        """
        Create video with tracking boxes and trajectory trails.
        
        Args:
            video_path: Path to original video
            tracking_data: Dictionary with tracking data per frame
            output_path: Path to save output video
            trajectory_length: Number of previous frames to show in trajectory
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Color palette for different players
        colors = plt.cm.get_cmap('tab20', 20)
        
        frame_count = 0
        history = {}  # Store recent positions for each player
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get tracking data for current frame
            current_tracks = tracking_data.get(frame_count, [])
            
            # Update history
            for track in current_tracks:
                player_id = track['id']
                center = track['center']
                
                if player_id not in history:
                    history[player_id] = []
                
                history[player_id].append(center)
                
                # Keep only recent history
                if len(history[player_id]) > trajectory_length:
                    history[player_id].pop(0)
            
            # Draw trajectories
            for player_id, positions in history.items():
                if len(positions) > 1:
                    # Get color for this player
                    color_idx = player_id % 20
                    color = colors(color_idx)
                    bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
                    
                    # Draw trajectory lines
                    for i in range(1, len(positions)):
                        pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                        pt2 = (int(positions[i][0]), int(positions[i][1]))
                        cv2.line(frame, pt1, pt2, bgr_color, 2)
            
            # Draw current detections
            for track in current_tracks:
                player_id = track['id']
                x1, y1, x2, y2 = map(int, track['bbox'])
                confidence = track['confidence']
                
                # Get color for this player
                color_idx = player_id % 20
                color = colors(color_idx)
                bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
                
                # Draw ID and confidence
                label = f"ID: {player_id} ({confidence:.2f})"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(frame, (x1, y1 - label_height - 10),
                            (x1 + label_width, y1), bgr_color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"Enhanced tracking video saved to {output_path}")
    
    @staticmethod
    def create_top_view_visualization(tracking_data, homography, 
                                     top_view_image_path=None,
                                     output_path='data/output/top_view_trajectories.mp4',
                                     fps=30):
        """
        Create top-view visualization of player movements.
        
        Args:
            tracking_data: Tracking data dictionary
            homography: HomographyTransformer instance with calculated H matrix
            top_view_image_path: Path to top-view background image
            output_path: Path to save output video
            fps: Frames per second for output video
        """
        if homography.homography_matrix is None:
            print("Homography matrix not calculated. Cannot create top-view visualization.")
            return
        
        # Load or create background
        if top_view_image_path and Path(top_view_image_path).exists():
            background = cv2.imread(top_view_image_path)
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        else:
            # Create green field background
            background = cv2.imread('data/right_half.png')
            print("Top-view image not found. Using data/right_half.png field background.")
        
        height, width = background.shape[:2]
        
        # Prepare video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Color palette
        colors = plt.cm.get_cmap('tab20', 20)
        
        # Track history for trails
        history = {}
        max_history = 60  # 2 seconds at 30 FPS
        
        # Find total frames
        total_frames = max(tracking_data.keys()) if tracking_data else 0
        
        for frame_idx in range(total_frames + 1):
            # Start with background
            frame = background.copy()
            
            # Get tracking data for this frame
            current_tracks = tracking_data.get(frame_idx, [])
            
            # Transform positions to top view
            for track in current_tracks:
                player_id = track['id']
                center = track['center']
                
                # Transform to top view
                try:
                    top_view_center = homography.transform_point(center)
                except:
                    continue
                
                # Update history
                if player_id not in history:
                    history[player_id] = []
                
                history[player_id].append((frame_idx, top_view_center))
                
                # Remove old history
                history[player_id] = [h for h in history[player_id] 
                                     if frame_idx - h[0] <= max_history]
            
            # Draw trajectories
            for player_id, positions in history.items():
                if len(positions) > 1:
                    color_idx = player_id % 20
                    color = colors(color_idx)
                    bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
                    
                    # Draw trajectory lines with fading
                    for i in range(1, len(positions)):
                        age = frame_idx - positions[i][0]
                        alpha = max(0.1, 1.0 - (age / max_history))
                        line_color = tuple(int(c * alpha) for c in bgr_color)
                        line_thickness = max(1, int(3 * alpha))
                        
                        pt1 = (int(positions[i-1][1][0]), int(positions[i-1][1][1]))
                        pt2 = (int(positions[i][1][0]), int(positions[i][1][1]))
                        cv2.line(frame, pt1, pt2, line_color, line_thickness)
            
            # Draw current positions
            for track in current_tracks:
                player_id = track['id']
                center = track['center']
                
                try:
                    top_view_center = homography.transform_point(center)
                except:
                    continue
                
                color_idx = player_id % 20
                color = colors(color_idx)
                bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
                
                # Draw player dot
                center_pos = (int(top_view_center[0]), int(top_view_center[1]))
                cv2.circle(frame, center_pos, 8, bgr_color, -1)
                cv2.circle(frame, center_pos, 8, (255, 255, 255), 1)
                
                # Draw player ID
                cv2.putText(frame, str(player_id), 
                          (center_pos[0] - 10, center_pos[1] - 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"Top-view visualization saved to {output_path}")
    
    @staticmethod
    def create_animated_plot(trajectories, output_path='data/output/animated_trajectories.gif'):
        """
        Create animated GIF of player trajectories.
        
        Args:
            trajectories: Trajectory data from TrajectoryAnalyzer
            output_path: Path to save animated GIF
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up plot
        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 720)
        ax.invert_yaxis()
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Player Trajectories Animation')
        ax.grid(True, alpha=0.3)
        
        # Color map
        cmap = plt.cm.get_cmap('tab20', len(trajectories))
        
        # Create line objects for each player
        lines = []
        for i, (player_id, traj) in enumerate(trajectories.items()):
            line, = ax.plot([], [], '-', color=cmap(i), linewidth=2, 
                          alpha=0.7, label=f'Player {player_id}')
            lines.append(line)
        
        ax.legend(loc='upper right', ncol=2)
        
        def init():
            for line in lines:
                line.set_data([], [])
            return lines
        
        def animate(frame):
            for idx, (player_id, traj) in enumerate(trajectories.items()):
                if frame < len(traj['x']):
                    lines[idx].set_data(traj['x'][:frame+1], traj['y'][:frame+1])
                else:
                    lines[idx].set_data(traj['x'], traj['y'])
            return lines
        
        # Create animation
        total_frames = max([len(traj['x']) for traj in trajectories.values()], default=0)
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=min(total_frames, 200),  # Limit frames for GIF size
                           interval=50, blit=True)
        
        # Save as GIF
        anim.save(output_path, writer='pillow', fps=20)
        plt.close()
        
        print(f"Animated plot saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    print("Visualization tools module loaded.")