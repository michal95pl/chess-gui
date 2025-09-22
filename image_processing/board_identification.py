import cv2

class BoardIdentification():

    def __init__(self, frame):
        self.frame = frame

    def identify(self, size=8):
        h, w, _ = self.frame.shape
        square_h = h // size
        square_w = w // size

        squares = []
        for y in range(size):
            row = []
            for x in range(size):

                y1, y2 = y * square_h, (y + 1) * square_h
                x1, x2 = x * square_w, (x + 1) * square_w

                square = self.frame[y1:y2, x1:x2]
                row.append(square)
            squares.append(row)
