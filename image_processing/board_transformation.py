import cv2
import numpy as np

from utils.logger import Logger


class BoardTransformation:

    def __get_color_centers(self, mask, frame, color=(0, 255, 0), min_area=50):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 8, color, 2)

        return centers

    def get_corners(self, red_points, green_points):
        if len(red_points) != 1 or len(green_points) != 3:
            raise Exception("Błąd: potrzebny 1 czerwony i 3 zielone punkty")

        red = red_points[0]

        distances = [np.linalg.norm(np.array(red) - np.array(g)) for g in green_points]
        farthest = green_points[np.argmax(distances)]  # bottom_right
        remaining = [g for g in green_points if not np.array_equal(g, farthest)]

        def side_of_line(p1, p2, pt):
            return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])

        sides = [side_of_line(red, farthest, pt) for pt in remaining]

        corners = {
            "top_left": red,
            "bottom_right": farthest,
        }

        if sides[0] > 0:
            corners["top_right"] = remaining[0]
            corners["bottom_left"] = remaining[1]
        else:
            corners["top_right"] = remaining[1]
            corners["bottom_left"] = remaining[0]

        return corners

    def draw_diagonals(self, frame, corners):
        top_left = corners["top_left"]
        bottom_right = corners["bottom_right"]
        top_right = corners["top_right"]
        bottom_left = corners["bottom_left"]

        cv2.line(frame, (int(top_left[0]), int(top_left[1])),
                 (int(bottom_right[0]), int(bottom_right[1])),
                 (255, 0, 0), 2)  # niebieska linia

        cv2.line(frame, (int(top_right[0]), int(top_right[1])),
                 (int(bottom_left[0]), int(bottom_left[1])),
                 (0, 255, 0), 2)  # zielona linia

        return frame

    def transform_to_square(self, frame, corners, size=800):
        src_pts = np.array([
            corners["top_left"],
            corners["top_right"],
            corners["bottom_right"],
            corners["bottom_left"]
        ], dtype=np.float32)

        dst_pts = np.array([
            [0, 0],
            [0, size - 1],
            [size - 1, size - 1],
            [size - 1, 0]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(frame, M, (size, size))

        return warped

    def crop_by_corners(self, frame, pattern_size=(7, 7), pad=91, output_size=800):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            raise Exception("Nie wykryto wzorca.")

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        pts = corners.reshape(-1, 2)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        h, w = frame.shape[:2]
        x1 = max(int(x_min) - pad, 0)
        y1 = max(int(y_min) - pad, 0)
        x2 = min(int(x_max) + pad, w)
        y2 = min(int(y_max) + pad, h)

        cropped = frame[y1:y2, x1:x2].copy()

        resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

        return resized

    def __rgb2hsv(self, rgb: tuple):
        color = np.uint8([[list(rgb)]])
        hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
        return tuple(hsv_color[0][0])

    def __init__(self, green_calibration: tuple, red_calibration: tuple, root):
        self.green_hsv_calibration = self.__rgb2hsv(green_calibration)
        self.red_hsv_calibration = self.__rgb2hsv(red_calibration)
        self.root = root

    def transform(self, frame):
        clean = frame.copy()
        hsv = cv2.cvtColor(clean, cv2.COLOR_RGB2HSV)

        # ---------- helpers ----------
        g = np.array(self.green_hsv_calibration, dtype=np.int32)
        r = np.array(self.red_hsv_calibration, dtype=np.int32)

        def make_bounds(centre, dH, dSV):
            lower = np.clip([centre[0] - dH, centre[1] - dSV, centre[2] - dSV],
                            [0, 0, 0], [179, 255, 255]).astype(np.uint8)
            upper = np.clip([centre[0] + dH, centre[1] + dSV, centre[2] + dSV],
                            [0, 0, 0], [179, 255, 255]).astype(np.uint8)
            return lower, upper

        # ---------- green mask ----------
        lo, hi = make_bounds(g, 15, 60)
        green_mask = cv2.inRange(hsv, lo, hi)

        # ---------- red mask (wrap-around) ----------
        lo1, hi1 = make_bounds(r, 10, 60)
        # if calibrated hue is <10 we need 0..calHue+10  AND  180-(10-calHue)..180
        if r[0] < 10:
            lo2 = np.array([180 - (10 - r[0]), lo1[1], lo1[2]], dtype=np.uint8)
            hi2 = np.array([179, hi1[1], hi1[2]], dtype=np.uint8)
            red_mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1),
                                      cv2.inRange(hsv, lo2, hi2))
        else:  # no wrap
            red_mask = cv2.inRange(hsv, lo1, hi1)

        # ---------- clean-up ----------
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # ---------- continue ----------
        green_centres = self.__get_color_centers(green_mask, clean, (0, 255, 0))
        red_centres = self.__get_color_centers(red_mask, clean, (0, 0, 255))

        # ---- rest of your code unchanged ----
        green_centers = self.__get_color_centers(green_mask, clean, (0, 255, 0))
        red_centers = self.__get_color_centers(red_mask, clean, (0, 0, 255))

        if green_centers is None or red_centers is None:
            raise Exception("Error: Green and Red centers are not found.")
            return frame
        else:
            print("Green and Red centers are found.")
            Logger.log("Green and Red centers are found.")

        corners = self.get_corners(red_centers, green_centers)
        if corners is None:
            raise Exception("Nie można wykonać transformacji – niewłaściwa liczba punktów.")

        #frame = self.transform_to_square(self.draw_diagonals(frame, corners), corners)
        frame = self.transform_to_square(frame, corners)
        cropped = self.crop_by_corners(frame)
        if cropped is not None:
            frame = cropped

        return frame


