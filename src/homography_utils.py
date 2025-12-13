import cv2
import numpy as np
import matplotlib.pyplot as plt

class HomographyTransformer:
    def __init__(self, top_view_path=None):
        """
        Initialize homography transformer for top-view projection.
        
        Args:
            ground_image_path: Path to ground image (camera view)
            top_view_path: Path to top-view reference image
        """
        self.src_points_manual = {
            0: (172, 319),
            1: (356, 320),
            2: (366, 318),
            3: (277, 281),
            4: (403, 315),
            5: (478, 254),
            6: (502, 294),
            7: (565, 280),
            8: (676, 292),
            9: (602, 315),
            10: (681, 507),
            11: (69, 468)
        }

        self.dst_points_manual = {
            5: (381, 202),
            6: (472, 267),
            7: (475, 215),
            8: (529, 183),
            9: (540, 261),
            10: (673, 393),
            11: (555, 531),
            1: (469, 367),
            4: (477, 343),
            2: (467, 359),
            0: (415, 448),
            3: (369, 362)
        }

        # self.known_distances = {
        #     'pitch_length': 20.12,  # meters
        #     'pitch_width': 3.05,    # meters
        #     'crease_length': 2.64,  # meters
        # }
        self.homography_matrix = None
        self.top_view_image = None
        self.ground_image_c = None
        self.top_view_image_c = None

        if top_view_path:
            self.top_view_image = cv2.imread(top_view_path)
    

        if self.src_points_manual is not None and self.dst_points_manual is not None:
            print("⚡ Using MANUAL points for homography")
            self.src_pts, self.dst_pts = self._prepare_manual_points(self.src_points_manual, self.dst_points_manual)
            self.homography_matrix = self.calculate_homography()
        else:
            raise ValueError("Source and destination points must be provided for homography calculation.")

    def _prepare_manual_points(self, src_dict, dst_dict):
        """Align manual dicts by key and convert to numpy arrays."""
        keys = sorted(src_dict.keys())

        src_pts = np.array([src_dict[k] for k in keys], dtype=np.float32)
        dst_pts = np.array([dst_dict[k] for k in keys], dtype=np.float32)

        print("Manual src_pts:\n", src_pts)
        print("Manual dst_pts:\n", dst_pts)
        return src_pts, dst_pts

    def calculate_homography(self):
        """
        Calculate homography matrix from source to destination points.
        
        Args:
            src_points: Source points (camera view)
            dst_points: Destination points (top view)
        
        Returns:
            Homography matrix
        """
        self.src_points = np.array(self.src_pts, dtype=np.float32)
        self.dst_points = np.array(self.dst_pts, dtype=np.float32)
        
        # Calculate homography using RANSAC for robustness
        self.homography_matrix, mask = cv2.findHomography(self.src_points, self.dst_points, cv2.RANSAC, 5.0)
        
        print(f"Homography matrix calculated (inliers: {mask.sum()}/{len(self.src_points)})")
        
        return self.homography_matrix
    
    def warp_image(self, image, output_size=None):
        """
        Warp image using homography.
        
        Args:
            image: Input image to warp
            output_size: Size of output image (width, height)
        
        Returns:
            Warped image
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not calculated.")
        
        if output_size is None and self.top_view_image_c is not None:
            output_size = (self.top_view_image_c.shape[1], self.top_view_image_c.shape[0])
        elif output_size is None:
            output_size = (image.shape[1], image.shape[0])
        
        warped = cv2.warpPerspective(image, self.homography_matrix, output_size)
        
        return warped
    
    def transform_point(self, point):
        """
        Transform a single point using homography.
        
        Args:
            point: (x, y) coordinate in source image
        
        Returns:
            Transformed point (x, y) in destination image
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not calculated. Call calculate_homography first.")
        
        point = np.array([point], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), self.homography_matrix)
        
        return transformed[0][0]
    
    def transform_points_batch(self, points):
        """
        Transform multiple points using homography.
        
        Args:
            points: List of (x, y) coordinates
        
        Returns:
            List of transformed points
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not calculated.")
        
        points = np.array(points, dtype=np.float32)
        transformed = cv2.perspectiveTransform(points.reshape(-1, 1, 2), self.homography_matrix)
        
        return transformed.reshape(-1, 2)
    
    def visualize_transform(self):
            """
            Visualize the homography transformation.
            
            Args:
                src_points: Source points (optional)
                dst_points: Destination points (optional)
            """
            if self.ground_image_c is None or self.top_view_image_c is None:
                print("Cannot visualize: Missing ground or top-view image")
                return
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot source image with points
            axes[0].imshow(self.ground_image_c)
            axes[0].set_title('Camera View (Source)')
            if self.src_points is not None:
                for i, (x, y) in enumerate(self.src_points):
                    axes[0].plot(x, y, 'ro', markersize=8)
                    axes[0].text(x, y, f'{i}', color='white', fontsize=12,
                            bbox=dict(facecolor='red', alpha=0.5))
            
            # Plot destination image with points
            axes[1].imshow(self.top_view_image_c)
            axes[1].set_title('Top View (Destination)')
            if self.dst_points is not None:
                for i, (x, y) in enumerate(self.dst_points):
                    axes[1].plot(x, y, 'go', markersize=8)
                    axes[1].text(x, y, f'{i}', color='white', fontsize=12,
                            bbox=dict(facecolor='green', alpha=0.5))
            
            # Plot warped image
            if self.homography_matrix is not None:
                warped = self.warp_image(self.ground_image_c)
                axes[2].imshow(warped)
                axes[2].set_title('Warped Camera View')
            
            plt.tight_layout()
            plt.savefig('data/output/homography_visualization.png', dpi=150)
            plt.show()
        
if __name__ == "__main__":
    # Example usage
    transformer = HomographyTransformer(
        top_view_path='data/top_view.png'
    )
    print("Homography Matrix:", transformer.homography_matrix)
    # Visualize
    transformer.visualize_transform()