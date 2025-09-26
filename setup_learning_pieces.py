from multiprocessing.reduction import duplicate

import cv2
import os
import glob

@staticmethod
def seperate(frame, flag):
    output_dir = "assets/chess_pieces"
    h, w, _ = frame.shape
    square_h = h // 8
    square_w = w // 8

    for y in range(8):
        for x in range(8):
            if y<=1 or y>=6:
                y1, y2 = y * square_h, (y + 1) * square_h
                x1, x2 = x * square_w, (x + 1) * square_w

                square = frame[y1:y2, x1:x2]

                color = None
                figure = None
                match y:
                    case 0 | 1:
                        color = "W"
                    case 6 | 7:
                        color = "B"
                if y == 1 or y == 6:
                    figure = "Pawn"
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

                os.makedirs(output_dir+f"/{color}_{figure}", exist_ok=True)
                filename = os.path.join(output_dir+f"/{color}_{figure}", f"square_{flag}_{y}_{x}.png")
                cv2.imwrite(filename, square)
                duplicate(square, output_dir+f"/{color}_{figure}", flag)

@staticmethod
def duplicate(square, filename, flag, steps = 20):
    h, w = square.shape[:2]
    center = (w // 2, h // 2)

    for x in range(steps):
        M = cv2.getRotationMatrix2D(center, x*(360/steps), 1.0)
        square = cv2.warpAffine(square, M, (w, h))
        for y in range(steps):
            alpha_darker = 1.0 - (y/steps)
            alpha_lighter = 1.0 + (y/steps)

            darker = cv2.convertScaleAbs(square, alpha=alpha_darker, beta=0)
            lighter = cv2.convertScaleAbs(square, alpha=alpha_lighter, beta=0)

            cv2.imwrite(os.path.join(filename, f"square_darker_{flag}_{x*(360/steps)}_{y}.png"), darker)
            cv2.imwrite(os.path.join(filename, f"square_lighter_{flag}_{x*(360/steps)}_{y}.png"), lighter)



#images = glob.glob("assets/chess_pieces/*/*.png")  # tylko PNG
#print(f"Liczba zdjęć PNG: {len(images)}")

seperate(cv2.imread('assets/ChessBoard1.png'), 1)
seperate(cv2.imread('assets/ChessBoard2.png'), 2)