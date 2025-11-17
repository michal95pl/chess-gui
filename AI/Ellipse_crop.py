import cv2
import albumentations as A
import numpy as np

class EllipseCrop(A.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0,
                 step_visualize=False, timewait=0,
                 min_area=50, max_area_ratio=0.8):
        super().__init__(always_apply, p)
        self.step_visualize = step_visualize
        self.timewait = timewait
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio

    def _show(self, name, img):
        if self.step_visualize:
            cv2.imshow(name, img)
            cv2.waitKey(self.timewait)

    def apply(self, img, **params):
        H, W = img.shape[:2]

        # ===== 1. PREPROCESS =====
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._show("Gray", gray)

        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._show("Threshold", th)

        edges = cv2.Canny(th, 100, 150)
        self._show("Edges", edges)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        self._show("Dilated", edges)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = H * W * self.max_area_ratio
        best_ellipse = None
        best_area = -1

        # ===== 2. SZUKANIE NAJWIĘKSZEJ ELIPSY =====
        for cnt in contours:
            if len(cnt) < 5:
                continue

            area = cv2.contourArea(cnt)
            if area < self.min_area or area > max_area:
                continue

            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (w, h), angle = ellipse

            if any([np.isnan(cx), np.isnan(cy), np.isnan(w), np.isnan(h)]):
                continue

            # współczynnik kołowości – odcinamy zbyt długie kształty
            ratio = max(w, h) / min(w, h)
            if ratio > 1.3:
                continue

            if area > best_area:
                best_area = area
                best_ellipse = ellipse

        # ===== 3. JEŚLI NIE MA ELIPSY → KOŁO NA CAŁE ZDJĘCIE =====
        if best_ellipse is None:
            cx, cy = W // 2, H // 2
            r = min(W, H) // 2
            best_ellipse = ((cx, cy), (2*r, 2*r), 0)

        # ===== 4. MASKOWANIE TŁA =====
        (cx, cy), (w, h), angle = best_ellipse

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(mask, best_ellipse, 255, -1)  # wypełniona elipsa
        self._show("Mask", mask)

        masked = cv2.bitwise_and(img, img, mask=mask)
        self._show("Masked", masked)

        # ===== 5. PRZYCINANIE, ABY ELIPSA ZAJĘŁA CAŁY OBRAZ =====
        w2, h2 = int(w/2), int(h/2)
        x1 = max(0, int(cx - w2))
        y1 = max(0, int(cy - h2))
        x2 = min(W, int(cx + w2))
        y2 = min(H, int(cy + h2))

        cropped = masked[y1:y2, x1:x2]
        self._show("Cropped", cropped)

        if self.step_visualize:
            cv2.destroyAllWindows()

        return cropped
