import cv2
import numpy as np
import matplotlib.pyplot as plt


def hist_modification(hist):
    # WIZUALIZACJA HISTOGRAMU

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

    dark_mode = np.argmax(hist[1:128])
    bright_mode = np.argmax(hist[128:]) + 128

    return (dark_mode + bright_mode) / 2, bright_mode, dark_mode

def get_threshold(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

    return hist_modification(hist)

def get_local_threshold(img, row, col, blocks=4):
    h, w = img.shape[:2]

        # rozmiar bloku 4x4
    bh, bw = h // blocks, w // blocks

        # ustalenie bloku (dla 8x8: 2 pola = 1 blok)
    br = row // (8 // blocks)
    bc = col // (8 // blocks)

        # wycinamy blok
    block = img[br * bh:(br + 1) * bh, bc * bw:(bc + 1) * bw]

        # grayscale
    if len(block.shape) == 3:
        gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    else:
        gray = block

    return get_threshold(gray)
