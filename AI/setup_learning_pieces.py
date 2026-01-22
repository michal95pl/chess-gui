import cv2
import os
import glob
import AI.Ellipse_crop
from image_processing import binary_treshold_finder

def cropp(square, filename):
    ellipse_crop = AI.Ellipse_crop.EllipseCrop()
    try:
        thr, bright_node, dark_node = binary_treshold_finder.get_threshold(square)
        square = cv2.resize(square, (100, 100), interpolation=cv2.INTER_AREA)
        square = ellipse_crop.apply(square, thr, bright_node, dark_node)
        cv2.imwrite(filename, square)
    except ValueError:
        print(f"Błąd przy cropie: {filename}, zapisuję oryginał")
        cv2.imwrite(filename, square)

def duplicate(square, filename, a, b, steps_r=24, steps_d=12):
    h, w = square.shape[:2]
    center = (w // 2, h // 2)

    for x in range(steps_r):
        M = cv2.getRotationMatrix2D(center, x*(360/steps_r), 1.0)
        rotated = cv2.warpAffine(square, M, (w, h))
        for y in range(steps_d):
            cropp(rotated, os.path.join(filename, f"square_{a}_{b}_{x*(360 / steps_r)}_{y}.png"))

def seperate(frame, name):
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
                filename = os.path.join(output_dir + f"/{figure}", f"{name}_{y}_{x}.png")
                cropp(square, filename)
                if figure in ["King", "Queen"]:
                    duplicate(square, output_dir + f"/{figure}", y, x, 24, 24)
                else:
                    duplicate(square, output_dir + f"/{figure}", y, x)
            else:
                continue

# Uruchomienie
seperate(cv2.imread('../assets/prepare/B1.png'),'Board1')
seperate(cv2.imread('../assets/prepare/B2.png'),'Board2')
seperate(cv2.imread('../assets/prepare/B3.png'),'Board3')

# Liczenie wygenerowanych obrazów
for folder in os.listdir('../assets/chess_pieces'):
    images = glob.glob(f"../assets/chess_pieces/{folder}/*.png")
    print(f"{folder}: {len(images)}")