from multiprocessing.reduction import duplicate

import cv2
import os
import glob

@staticmethod
def seperate(frame, flag):
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
                match y:
                    case 0 | 1:
                        color = "W"
                    case 6 | 7:
                        color = "B"

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

                if flag==1:
                    if y==0 and x==3:
                        figure = "Queen"
                    if y==0 and x==4:
                        figure = "King"
                    if y==7 and x==3:
                        figure = "King"
                    if y==7 and x==4:
                        figure = "Queen"
                elif flag==2:
                    if y==0 and x==3:
                        figure = "King"
                    if y==0 and x==4:
                        figure = "Queen"
                    if y==7 and x==3:
                        figure = "Queen"
                    if y==7 and x==4:
                        figure = "King"

                if figure=="King" or figure=="Queen":
                    os.makedirs(output_dir + f"/{color}_{figure}", exist_ok=True)
                    filename = os.path.join(output_dir + f"/{color}_{figure}", f"square_{flag}_{y}_{x}.png")
                    cv2.imwrite(filename, square)
                    duplicate(square, output_dir + f"/{color}_{figure}", flag, y, x, 8, 16)
                else:
                    os.makedirs(output_dir + f"/{color}_{figure}", exist_ok=True)
                    filename = os.path.join(output_dir + f"/{color}_{figure}", f"square_{flag}_{y}_{x}.png")
                    cv2.imwrite(filename, square)
                    duplicate(square, output_dir + f"/{color}_{figure}", flag, y, x)

            else:
                os.makedirs(output_dir + f"/Empty", exist_ok=True)
                filename = os.path.join(output_dir + f"/Empty", f"square_{flag}_{y}_{x}.png")
                cv2.imwrite(filename, square)
                duplicate(square, output_dir + f"/Empty", flag, y, x, 1, 4)

@staticmethod
def duplicate(square, filename, flag, a, b, steps_r= 8, steps_h=8):
    h, w = square.shape[:2]
    center = (w // 2, h // 2)

    for x in range(steps_r):
        M = cv2.getRotationMatrix2D(center, x*(360/steps_r), 1.0)
        square = cv2.warpAffine(square, M, (w, h))
        for y in range(steps_h):
            alpha_darker = 1.0 - (y / steps_h)
            alpha_lighter = 1.0 + (y / steps_h)

            darker = cv2.convertScaleAbs(square, alpha=alpha_darker, beta=0)
            lighter = cv2.convertScaleAbs(square, alpha=alpha_lighter, beta=0)

            cv2.imwrite(os.path.join(filename, f"square_darker_{flag}_{a}_{b}_{x*(360 / steps_r)}_{y}.png"), darker)
            cv2.imwrite(os.path.join(filename, f"square_lighter_{flag}_{a}_{b}_{x*(360 / steps_r)}_{y}.png"), lighter)

seperate(cv2.imread('../assets/ChessBoard1.png'), 1)
seperate(cv2.imread('../assets/ChessBoard2.png'), 2)

for folder in os.listdir('../assets/chess_pieces'):
        images = glob.glob(f"../assets/chess_pieces/{folder}/*.png")
        print(f"{folder}: {len(images)}")