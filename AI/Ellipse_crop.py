import cv2
import albumentations as A
import numpy as np
from image_processing import binary_treshold_finder

class EllipseCrop(A.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0,
                 step_visualize=False, timewait=500, scale = 1.1):
        super().__init__(always_apply, p)
        self.step_visualize = step_visualize
        self.timewait = timewait
        self.scale = scale

    def _show(self, name, img):
        if self.step_visualize:
            cv2.imshow(name, img)
            cv2.waitKey(self.timewait)

    def find(self, img, thr, bright_node, dark_node, debug=False, margin=1):
        # Konwersja do szarości
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Progowanie
        _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

        # Edge detection (opcjonalnie)
        edges = cv2.Canny(th, bright_node-10, dark_node+10)

        # Znalezienie konturów
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtruj kontury blisko krawędzi
        H, W = img.shape[:2]
        valid_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if x > margin and y > margin and x + w < W - margin and y + h < H - margin:
                valid_contours.append(cnt)

        if debug:
            vis = img.copy()
            cv2.drawContours(vis, valid_contours, -1, (0, 255, 0), 1)
            cv2.imshow("Contours", vis)
            cv2.waitKey(0)
            cv2.destroyWindow("Contours")

        return len(valid_contours) > 0

    def apply(self, img, **params):
        H, W = img.shape[:2]
        vis = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thr, b_node, d_node = binary_treshold_finder.get_threshold(img)
        _, img = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)

        # ===== 1. PREPROCESS =====
        self._show("Gray", img)

        edges = cv2.Canny(img, b_node, d_node)
        self._show("Edges", edges)

        kernel = np.ones((1, 1), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        self._show("Dilated", edges)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_ellipse = None
        best_score = -1  # score = area / ratio

        for cnt in contours:
            if len(cnt) < 5 or cv2.contourArea(cnt) <= 0:
                continue

            area = cv2.contourArea(cnt)

            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (w, h), angle = ellipse

            if any([np.isnan(cx), np.isnan(cy), np.isnan(w), np.isnan(h)]):
                continue

            if w <= 0 or h <= 0:
                continue

            ratio = max(w, h) / min(w, h)
            if ratio > 1.5:
                continue

            score = area / ratio
            if best_ellipse is None:
                best_ellipse = ellipse
                best_score = score
            elif score > best_score:
                best_ellipse = ellipse
                best_score = score

            # ===== Rysowanie elipsy =====
            cv2.ellipse(vis, ellipse, (0, 255, 0), 2)  # zielona elipsa
            self._show("Ellipse", vis)

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
