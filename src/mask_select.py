import cv2
import numpy as np

class OvalROISelector:
    def __init__(self, frame, display_width=1280):
        self.frame = frame
        self.clone = frame.copy()

        h, w = frame.shape[:2]

        # Display scaling for visualization
        self.display_width = display_width
        self.scale = display_width / w
        self.display_height = int(h * self.scale)

        # Ellipse parameters (original resolution)
        self.center = [w // 2, h // 2]
        self.axes = [w // 4, h // 3]
        self.angle = 0.0

        # State flags
        self.dragging = False
        self.resizing = False
        self.waiting_for_pivot = False

        # Rotation pivot
        self.rotation_pivot = None

        # Rotation keyboard
        self.rotation_step = 0.5  # degrees per keypress

        cv2.namedWindow("Select Pitch ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Pitch ROI", self.display_width, self.display_height)
        cv2.setMouseCallback("Select Pitch ROI", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        ox = int(x / self.scale)
        oy = int(y / self.scale)

        # Set rotation pivot
        if event == cv2.EVENT_LBUTTONDOWN and self.waiting_for_pivot:
            self.rotation_pivot = (ox, oy)
            self.waiting_for_pivot = False
            return

        # Move / Resize
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_inside_ellipse(ox, oy):
                self.dragging = True
                self.prev = (ox, oy)
            else:
                self.resizing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                dx = ox - self.prev[0]
                dy = oy - self.prev[1]
                self.center[0] += dx
                self.center[1] += dy
                self.prev = (ox, oy)
            elif self.resizing:
                self.axes[0] = max(10, abs(ox - self.center[0]))
                self.axes[1] = max(10, abs(oy - self.center[1]))

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.resizing = False

    def is_inside_ellipse(self, x, y):
        dx = (x - self.center[0]) / self.axes[0]
        dy = (y - self.center[1]) / self.axes[1]
        return dx * dx + dy * dy <= 1

    def get_mask(self):
        while True:
            # Start with original frame
            temp2 = self.clone.copy()

            # Draw ellipse on separate mask
            ellipse_mask = np.zeros_like(self.clone, dtype=np.uint8)
            cv2.ellipse(
                ellipse_mask,
                tuple(self.center),
                tuple(self.axes),
                0, 0, 360,
                (0, 0, 255),
                2
            )

            # Rotate ellipse around pivot if set
            if self.rotation_pivot is not None:
                M = cv2.getRotationMatrix2D(self.rotation_pivot, self.angle, 1.0)
                rotated_mask = cv2.warpAffine(ellipse_mask, M, (ellipse_mask.shape[1], ellipse_mask.shape[0]))
            else:
                rotated_mask = ellipse_mask

            # Overlay rotated ellipse on original frame
            temp = cv2.addWeighted(temp2, 1.0, rotated_mask, 1.0, 0)

            # Draw pivot for visualization
            if self.rotation_pivot is not None:
                cv2.circle(temp, self.rotation_pivot, 5, (0, 0, 255), -1)

            # Resize for display
            display = cv2.resize(temp, (self.display_width, self.display_height))
            cv2.putText(
                display,
                "Drag: move | Drag outside: resize | R: pivot | Q/E: rotate | ENTER",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.imshow("Select Pitch ROI", display)

            key = cv2.waitKey(1)

            # Set pivot
            if key == ord('r'):
                self.waiting_for_pivot = True
            # Rotate
            elif key == ord('q'):
                self.angle = (self.angle - self.rotation_step) % 360
            elif key == ord('e'):
                self.angle = (self.angle + self.rotation_step) % 360
            elif key == 13:  # ENTER
                break
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None

        # Create final mask at original resolution
        mask2 = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(
            mask2,
            tuple(self.center),
            tuple(self.axes),
            0, 0, 360,
            255,
            -1
        )
        if self.rotation_pivot is not None:
            M = cv2.getRotationMatrix2D(self.rotation_pivot, self.angle, 1.0)
            mask = cv2.warpAffine(mask2, M, (mask2.shape[1], mask2.shape[0]))
        else:
            mask = mask2

        cv2.destroyAllWindows()
        self.angle = float(self.angle)
        self.center = tuple(map(int, self.center))
        self.axes = tuple(map(int, self.axes))

        return mask


# -------------------------
# EXAMPLE USAGE
# -------------------------
if __name__ == "__main__":
    frame = cv2.imread("frame.jpg")  # Replace with your video frame
    selector = OvalROISelector(frame)
    mask = selector.get_mask()

    if mask is not None:
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
