import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import DBSCAN
import json
from pathlib import Path

class TrajectoryAnalyzer:
    def __init__(self, tracking_data_path=None):
        """
        Initialize trajectory analyzer.
        
        Args:
            tracking_data_path: Path to saved tracking data JSON
        """
        self.trajectories = {}
        self.heatmap = None
        
        if tracking_data_path:
            self.load_tracking_data(tracking_data_path)
    
    def load_tracking_data(self, filepath):
        """
        Load tracking data from JSON file.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.trajectories = {}
        for track_id, history in data['tracks_history'].items():
            self.trajectories[track_id] = {
                'x': [point['center'][0] for point in history],
                'y': [point['center'][1] for point in history],
                'frames': [point['frame'] for point in history]
            }
        
        print(f"Loaded {len(self.trajectories)} player trajectories")
    
    def calculate_heatmap(self, grid_size=(50, 50), video_size=(1280, 720)):
        """
        Calculate heatmap of player positions.
        
        Args:
            grid_size: Number of grid cells (width, height)
            video_size: Video dimensions
        
        Returns:
            Heatmap array
        """
        all_points = []
        for track_id, traj in self.trajectories.items():
            for x, y in zip(traj['x'], traj['y']):
                all_points.append([x, y])
        
        if not all_points:
            return None
        
        all_points = np.array(all_points)
        
        # Create grid
        grid_x = np.linspace(0, video_size[0], grid_size[0])
        grid_y = np.linspace(0, video_size[1], grid_size[1])
        
        # Calculate 2D histogram
        self.heatmap, x_edges, y_edges = np.histogram2d(
            all_points[:, 0], all_points[:, 1],
            bins=[grid_x, grid_y]
        )
        
        return self.heatmap
    
    def plot_trajectories(self, output_path='data/output/trajectories.png'):
        """
        Plot all player trajectories.
        
        Args:
            output_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Generate distinct colors for each trajectory
        cmap = plt.cm.get_cmap('tab20', len(self.trajectories))
        
        for i, (track_id, traj) in enumerate(self.trajectories.items()):
            color = cmap(i)
            ax.plot(traj['x'], traj['y'], '-', color=color, alpha=0.7, linewidth=2,
                   label=f'Player {track_id}')
            # Plot start and end points
            ax.plot(traj['x'][0], traj['y'][0], 'o', color=color, markersize=8)
            ax.plot(traj['x'][-1], traj['y'][-1], 's', color=color, markersize=8)
        
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title('Player Trajectories')
        ax.invert_yaxis()  # Invert y-axis for image coordinates
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.show()
        
        return fig
    
    def plot_heatmap(self, video_size=(1280, 720), output_path='data/output/heatmap.png'):
        """
        Plot heatmap of player positions.
        
        Args:
            video_size: Video dimensions
            output_path: Path to save the plot
        """
        if self.heatmap is None:
            self.calculate_heatmap(video_size=video_size)
        
        if self.heatmap is None:
            print("No data for heatmap")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot heatmap
        im = ax.imshow(self.heatmap.T, origin='lower',
                      extent=[0, video_size[0], 0, video_size[1]],
                      cmap='hot', alpha=0.7)
        
        # Overlay trajectories
        for track_id, traj in self.trajectories.items():
            ax.plot(traj['x'], traj['y'], '-', color='cyan', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title('Player Position Heatmap with Trajectories')
        ax.invert_yaxis()  # Invert y-axis for image coordinates
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Position Frequency')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.show()
        
        return fig
    
    def calculate_movement_stats(self):
        """
        Calculate movement statistics for each player.
        
        Returns:
            Dictionary with movement statistics
        """
        stats_dict = {}
        
        for track_id, traj in self.trajectories.items():
            if len(traj['x']) < 2:
                continue
            
            # Calculate distances between consecutive points
            distances = []
            speeds = []
            
            for i in range(1, len(traj['x'])):
                dx = traj['x'][i] - traj['x'][i-1]
                dy = traj['y'][i] - traj['y'][i-1]
                distance = np.sqrt(dx**2 + dy**2)
                distances.append(distance)
                
                # Assuming 30 FPS, convert to pixels per second
                speed = distance * 30  # pixels per second
                speeds.append(speed)
            
            if distances:
                stats_dict[track_id] = {
                    'total_distance': sum(distances),
                    'avg_speed': np.mean(speeds),
                    'max_speed': np.max(speeds),
                    'min_speed': np.min(speeds),
                    'std_speed': np.std(speeds),
                    'num_points': len(traj['x']),
                    'area_covered': self.calculate_area_covered(traj)
                }
        
        return stats_dict
    
    def calculate_area_covered(self, trajectory, grid_size=10):
        """
        Calculate approximate area covered by a player.
        
        Args:
            trajectory: Player trajectory data
            grid_size: Grid cell size in pixels
        
        Returns:
            Area in grid units
        """
        if not trajectory['x']:
            return 0
        
        # Create grid
        min_x, max_x = min(trajectory['x']), max(trajectory['x'])
        min_y, max_y = min(trajectory['y']), max(trajectory['y'])
        
        # Create binary grid
        x_bins = int((max_x - min_x) / grid_size) + 1
        y_bins = int((max_y - min_y) / grid_size) + 1
        
        if x_bins <= 0 or y_bins <= 0:
            return 0
        
        grid = np.zeros((x_bins, y_bins), dtype=bool)
        
        # Mark visited cells
        for x, y in zip(trajectory['x'], trajectory['y']):
            i = int((x - min_x) / grid_size)
            j = int((y - min_y) / grid_size)
            if 0 <= i < x_bins and 0 <= j < y_bins:
                grid[i, j] = True
        
        # Calculate area (number of visited cells * cell area)
        area = np.sum(grid) * (grid_size ** 2)
        
        return area
    
    def detect_clusters(self, epsilon=50, min_samples=2):
        """
        Detect spatial clusters of players using DBSCAN.
        
        Args:
            epsilon: Maximum distance between points in a cluster
            min_samples: Minimum number of points to form a cluster
        
        Returns:
            Cluster labels and information
        """
        all_points = []
        for track_id, traj in self.trajectories.items():
            # Use average position for each player
            if traj['x'] and traj['y']:
                avg_x = np.mean(traj['x'])
                avg_y = np.mean(traj['y'])
                all_points.append([avg_x, avg_y])
        
        if not all_points:
            return None, None
        
        all_points = np.array(all_points)
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = dbscan.fit_predict(all_points)
        
        # Count clusters (excluding noise points labeled as -1)
        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        n_noise = list(labels).count(-1)
        
        print(f"Detected {n_clusters} clusters with {n_noise} noise points")
        
        return labels, all_points
    
    def export_analysis_report(self, output_path='data/output/analysis_report.json'):
        """
        Export comprehensive analysis report.
        
        Args:
            output_path: Path to save the report
        """
        report = {
            'num_players': len(self.trajectories),
            'movement_stats': self.calculate_movement_stats(),
            'trajectory_lengths': {
                track_id: len(traj['x'])
                for track_id, traj in self.trajectories.items()
            },
            'clustering': {}
        }
        
        # Add clustering info
        labels, points = self.detect_clusters()
        if labels is not None:
            report['clustering'] = {
                'labels': labels.tolist(),
                'points': points.tolist() if points is not None else [],
                'num_clusters': len(set(labels)) - (1 if -1 in labels else 0)
            }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))
        
        print(f"Analysis report saved to {output_path}")
        
        return report

if __name__ == "__main__":
    # Example usage
    analyzer = TrajectoryAnalyzer('data/output/tracking_data.json')
    
    # Generate visualizations
    analyzer.plot_trajectories()
    analyzer.plot_heatmap(video_size=(1280, 720))
    
    # Export analysis
    analyzer.export_analysis_report()