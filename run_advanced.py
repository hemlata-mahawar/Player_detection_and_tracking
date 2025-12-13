"""
Advanced pipeline with all features enabled.
"""

import sys
from pathlib import Path
import argparse


from src.detect_and_track import CricketPlayerTracker
from src.trajectory_analyzer import TrajectoryAnalyzer
from src.visualization import VisualizationTools
from src.homography_utils import HomographyTransformer

def main():
    parser = argparse.ArgumentParser(description='Advanced Cricket Player Tracking Pipeline')
    parser.add_argument('--video', type=str, help='Path to input video', default='data/v3_one_min.mp4')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--ground', type=str, help='Path to ground image for homography')
    parser.add_argument('--top-view', type=str, help='Path to top-view image')
    parser.add_argument('--show-mask', type=bool, help='Whether to show pitch mask during processing', default=True)
    
    args = parser.parse_args()
    
    # Set default paths for ground and top-view images if not provided
    if args.ground is None:
        args.ground = 'data/ground.png'
    if args.top_view is None:
        args.top_view = 'data/top_view.png'
    
    # Check if files exist
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1
    

    ground_image=args.ground if Path(args.ground).exists() else None
    top_view_image=args.top_view if Path(args.top_view).exists() else None

    output_dir = Path(args.output)
    output_dir = output_dir / f"{video_path.stem}_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Advanced Cricket Player Tracking Pipeline")
    print("=" * 60)
    
    # Initialize tracker
    print("\n[1/5] Initializing tracker...")
    tracker = CricketPlayerTracker()
    
    # Run detection and tracking
    print("\n[2/5] Detecting and tracking players...")
    output_video = output_dir / f"{Path(video_path).stem}_tracked.mp4"
    
    show_mask = args.show_mask
    results = tracker.detect_and_track(
        video_path=video_path,
        output_path=str(output_video),
        show_video=False,
        save_output=True,
        show_mask=show_mask
    )
    
    tracking_json = output_dir / "tracking_data.json"
    tracker.save_tracking_data(str(tracking_json))
    
    # Trajectory analysis
    print("\n[3/5] Analyzing trajectories...")
    analyzer = TrajectoryAnalyzer(str(tracking_json))
    
    # Generate all visualizations
    analyzer.plot_trajectories(output_path=output_dir / "trajectories.png")
    analyzer.plot_heatmap(
        video_size=(results['video_info']['width'], results['video_info']['height']),
        output_path=output_dir / "heatmap.png"
    )
    # Export comprehensive report
    print("\n[4/5] Generating comprehensive report...")
    analyzer.export_analysis_report(output_path=output_dir / "full_report.json")
    
    
    # Enhanced video with trajectories
    enhanced_video = output_dir / "enhanced_trajectories.mp4"
    VisualizationTools.create_tracking_video_with_trajectories(
        video_path=video_path,
        tracking_data=results['tracking_data'],
        output_path=str(enhanced_video)
    )
    
    # Top-view projection
    # if ground_image and top_view_image:
    if top_view_image:
        print("\n[5/5] Creating top-view projection...")
        
        homography = HomographyTransformer(
            top_view_path=top_view_image
        )        
        # Create top-view video
        top_view_video = output_dir / "top_view.mp4"
        VisualizationTools.create_top_view_visualization(
            tracking_data=results['tracking_data'],
            homography=homography,
            # top_view_image_path=ground_top_view_image,
            output_path=str(top_view_video),
            fps=results['video_info']['fps']
        )
        
        # Save homography visualization
        homography.visualize_transform()
    
    # Create animated GIF if video is not too long
    total_frames = results['video_info']['total_frames']
    if total_frames < 300:
        VisualizationTools.create_animated_plot(
            analyzer.trajectories,
            output_path=output_dir / "animated_trajectories.gif"
        )
    
    print("\n" + "=" * 60)
    print("✅ Advanced pipeline completed!")
    print("=" * 60)
    
    # Print file summary
    print("\n📁 Generated files:")
    for file in sorted(output_dir.glob("*")):
        if file.is_file():
            size = file.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f} MB"
            else:
                size_str = f"{size/1024:.1f} KB"
            print(f"  • {file.name:30} ({size_str})")
    
    return results

if __name__ == "__main__":
    sys.exit(main())