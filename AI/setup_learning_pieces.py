import cv2
import os
import glob
import AI.Ellipse_crop
import numpy as np
import matplotlib.pyplot as plt


def get_threshold(square):
    hist = cv2.calcHist([square], [0], None, [256], [0, 256]).flatten()

    # dwa największe mody
    dark_mode = np.argmax(hist[1:128])
    bright_mode = np.argmax(hist[128:]) + 128

    thr = (dark_mode + bright_mode) / 2

    #WIZUALIZACJA HISTOGRAMU

    # plt.figure(figsize=(8, 4))
    # plt.plot(hist, color='black')
    # plt.axvline(dark_mode, color='blue', linestyle='--', label=f'Dark mode: {dark_mode}')
    # plt.axvline(bright_mode, color='red', linestyle='--', label=f'Bright mode: {bright_mode}')
    # plt.axvline(thr, color='green', linestyle='-', label=f'Threshold: {int(thr)}')
    # plt.title("Histogram grayscale")
    # plt.xlabel("Intensywność piksela")
    # plt.ylabel("Liczba pikseli")
    # plt.legend()
    # plt.show()
    # plt.pause(1)

    return thr

def cropp(square, filename):
    ellipse_crop = AI.Ellipse_crop.EllipseCrop()
    try:
        square = ellipse_crop.apply(square)
        _, square = cv2.threshold(square, get_threshold(square), 255, cv2.THRESH_BINARY)
        cv2.imwrite(filename, square)
    except ValueError:
        print(f"Błąd przy cropie: {filename}, zapisuję oryginał")
        cv2.imwrite(filename, square)

def duplicate(square, filename, a, b, steps_r=4, steps_d=20):
    h, w = square.shape[:2]
    center = (w // 2, h // 2)

    for x in range(steps_r):
        M = cv2.getRotationMatrix2D(center, x*(360/steps_r), 1.0)
        rotated = cv2.warpAffine(square, M, (w, h))
        for y in range(steps_d):
            cropp(rotated, os.path.join(filename, f"square_{a}_{b}_{x*(360 / steps_r)}_{y}.png"))

def seperate(frame):
    output_dir = "../assets/chess_pieces"
    h, w, _ = frame.shape
    square_h = h // 8
    square_w = w // 8

    for y in range(8):
        for x in range(8):
            y1, y2 = y * square_h, (y + 1) * square_h
            x1, x2 = x * square_w, (x + 1) * square_w
            square = frame[y1:y2, x1:x2]

            if y<=1 or y>=6:
                color = None
                figure = None

                if (y == 1 or y == 6) and (x == 0 or x == 1):
                    figure = "Pawn"
                elif (y==1 or y==6) and x>1:
                    continue
                else:
                    match x:
                        case 0 | 7:
                            figure = "Rook"
                        case 1 | 6:
                            figure = "Knight"
                        case 2 | 5:
                            figure = "Bishop"
                        case 3:
                            figure = "King"
                        case 4:
                            figure = "Queen"

                os.makedirs(output_dir + f"/{figure}", exist_ok=True)
                filename = os.path.join(output_dir + f"/{figure}", f"square_{y}_{x}.png")
                cropp(square, filename)
                if figure in ["King", "Queen"]:
                    duplicate(square, output_dir + f"/{figure}", y, x, 4, 40)
                else:
                    duplicate(square, output_dir + f"/{figure}", y, x)
            else:
                continue

# Uruchomienie
seperate(cv2.imread('../assets/ChessBoard1.png'))
seperate(cv2.imread('../assets/ChessBoard2.png'))

# Liczenie wygenerowanych obrazów
for folder in os.listdir('../assets/chess_pieces'):
    images = glob.glob(f"../assets/chess_pieces/{folder}/*.png")
    print(f"{folder}: {len(images)}")
