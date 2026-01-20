import cv2
import albumentations as A
import numpy as np

class EllipseCrop(A.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0,
                  timewait=500, scale = 1.1):
        super().__init__(always_apply, p)
        self.step_visualize = None
        self.timewait = timewait
        self.scale = scale

    def _show(self, name, img):
        if self.step_visualize:
            cv2.imshow(name, img)
            #cv2.imwrite(f"image_processing/steps/identification_{name}.png", img)
            cv2.waitKey(self.timewait)

    def detect_ellipse(self, img, thr, bright_node, dark_node, step_visualize=None):
        self.step_visualize = step_visualize
        H, W = img.shape[:2]
        vis = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

        self._show("Gray1", th)

        edges = cv2.Canny(th, bright_node, dark_node)
        self._show("Edges2", edges)

        kernel = np.ones((1, 1), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        self._show("Dilated3", edges)

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

            if ellipse_area < H * W * 0.2 or ellipse_area > H * W * 0.7:
                continue

            score = area / ratio

            if best_ellipse is None or score > best_score:
                best_ellipse = ellipse
                best_score = score

            cv2.ellipse(vis, ellipse, (0, 255, 0), 2)
            self._show("Ellipse4", vis)

        if self.step_visualize:
            cv2.destroyAllWindows()

        return best_ellipse

    def find(self, img, thr, bright_node, dark_node, step_visualize=False):
        H, W = img.shape[:2]

        # 1. Wycinamy środek (80%) - usuwamy krawędzie szachownicy
        margin_h = int(H * 0.1)  # 10% marginesu z góry i dołu
        margin_w = int(W * 0.1)  # 10% marginesu z boków
        center_roi = img[margin_h:H - margin_h, margin_w:W - margin_w]

        if len(center_roi.shape) == 3:
            gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = center_roi

        # 2. Binarizacja wyciętego środka
        _, binary = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

        # Obliczamy procent wypełnienia (tylko dla środka)
        h_roi, w_roi = binary.shape[:2]
        white_ratio = cv2.countNonZero(binary) / (h_roi * w_roi)

        # Jeśli pole jest jednolite (czarne lub białe) po wycięciu brzegów -> PUSTE
        # Zwiększyłem margines tolerancji, bo środek jest bardziej stabilny
        if white_ratio > 0.98 or white_ratio < 0.02:
            return False

        # 3. Sprawdzenie krawędzi (Canny) na środku pola
        # Robimy to na skali szarości, nie na binarnym (lepiej dla białe-na-białym)
        edges = cv2.Canny(gray, int(bright_node * 0.4), int(dark_node * 0.4))
        edge_density = np.sum(edges > 0) / (h_roi * w_roi)

        # Pokazujemy debug jeśli trzeba
        if step_visualize:
            cv2.imshow("ROI_Edges", edges)

        # Jeśli brak krawędzi w centrum -> PUSTE
        if edge_density < 0.01:
            return False

        # 4. Jeśli przeszedł testy "środka", uruchamiamy pełną detekcję elipsy na całym polu
        ellipse = self.detect_ellipse(img, thr, bright_node, dark_node, step_visualize=step_visualize)
        return ellipse is not None

    def apply(self, img, thr, bright_node, dark_node, step_visualize=False):
        H, W = img.shape[:2]
        vis = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        best_ellipse = self.detect_ellipse(vis, thr, bright_node, dark_node, step_visualize=step_visualize)

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

        self._show("Mask5", mask)
        masked = cv2.bitwise_and(img, img, mask=mask)
        self._show("Masked6", masked)

        # ===== 5. PRZYCINANIE, ABY ELIPSA ZAJĘŁA CAŁY OBRAZ =====
        w2, h2 = int(w_scaled/2), int(h_scaled/2)
        x1 = max(0, int(cx - w2))
        y1 = max(0, int(cy - h2))
        x2 = min(W, int(cx + w2))
        y2 = min(H, int(cy + h2))

        cropped = masked[y1:y2, x1:x2]
        self._show("Cropped7", cropped)

        if self.step_visualize:
            cv2.destroyAllWindows()

        return cropped
