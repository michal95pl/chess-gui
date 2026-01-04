import cv2
import albumentations as A
import numpy as np
from image_processing import binary_treshold_finder

class EllipseCrop(A.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0,
                 step_visualize=False, timewait=100, scale = 1.1):
        super().__init__(always_apply, p)
        self.step_visualize = step_visualize
        self.timewait = timewait
        self.scale = scale

    def _show(self, name, img):
        if self.step_visualize:
            cv2.imshow(name, img)
            cv2.waitKey(self.timewait)

    def detect_ellipse(self, img, thr, bright_node, dark_node):
        H, W = img.shape[:2]
        vis = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

        self._show("Gray", th)

        edges = cv2.Canny(th, bright_node, dark_node)
        self._show("Edges", edges)

        kernel = np.ones((1, 1), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        self._show("Dilated", edges)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_ellipse = None
        best_score = -1

        for cnt in contours:
            if len(cnt) < 5 or cv2.contourArea(cnt) <= 0:
                continue

            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (w, h), angle = ellipse
            area = cv2.contourArea(cnt)
            ellipse_area = np.pi * (w / 2) * (h / 2)

            if any([np.isnan(cx), np.isnan(cy), np.isnan(w), np.isnan(h)]):
                continue

            if w <= 0 or h <= 0:
                continue

            ratio = max(w, h) / min(w, h)
            if ratio > 1.5:
                continue

            if ellipse_area < H * W * 0.3 or ellipse_area > H * W * 0.6:
                continue

            score = area / ratio

            if best_ellipse is None or score > best_score:
                best_ellipse = ellipse
                best_score = score

            cv2.ellipse(vis, ellipse, (0, 255, 0), 2)
            self._show("Ellipse", vis)

        if self.step_visualize:
            cv2.destroyAllWindows()

        return best_ellipse

    def find(self, img, thr, bright_node, dark_node):
        ellipse = self.detect_ellipse(img, thr, bright_node, dark_node)
        return ellipse is not None

    def apply(self, img, thr, bright_node, dark_node):
        H, W = img.shape[:2]
        print(H,W)
        vis = img.copy()

        best_ellipse = self.detect_ellipse(vis, thr, bright_node, dark_node)

        # ===== 3. JEŚLI NIE MA ELIPSY → KOŁO NA CAŁE ZDJĘCIE =====
        if best_ellipse is None:
            cx, cy = W // 2, H // 2
            r = min(W, H) // 2
            best_ellipse = ((cx, cy), (2*r, 2*r), 0)

        # ===== 4. MASKOWANIE TŁA =====
        (cx, cy), (w, h), angle = best_ellipse
        w_scaled = w * self.scale
        h_scaled = h * self.scale
        best_ellipse_scaled = ((cx, cy), (w_scaled, h_scaled), angle)
        mask = np.zeros((H, W), dtype=np.uint8)

        # rysujemy elipsę tylko jeśli wymiary są poprawne
        if w > 0 and h > 0:
            cv2.ellipse(mask, best_ellipse_scaled, 255, -1)
        else:
            # brak poprawnej elipsy → rysujemy okrąg na całe zdjęcie
            r = min(W, H) // 2
            cv2.circle(mask, (W // 2, H // 2), r, 255, -1)

        self._show("Mask", mask)
        masked = cv2.bitwise_and(img, img, mask=mask)
        self._show("Masked", masked)

        # ===== 5. PRZYCINANIE, ABY ELIPSA ZAJĘŁA CAŁY OBRAZ =====
        w2, h2 = int(w_scaled/2), int(h_scaled/2)
        x1 = max(0, int(cx - w2))
        y1 = max(0, int(cy - h2))
        x2 = min(W, int(cx + w2))
        y2 = min(H, int(cy + h2))

        cropped = masked[y1:y2, x1:x2]
        self._show("Cropped", cropped)

        if self.step_visualize:
            cv2.destroyAllWindows()

        return cropped
