from Ellipse_crop import EllipseCrop
import cv2

img_path = r"../assets/chess_pieces/B_King/square_0_3.png"
img = cv2.imread(img_path)[:,:,::-1]  # BGR → RGB

# włącz wizualizację krok po kroku
try:
    crop = EllipseCrop(step_visualize=True, timewait=1200)(image=img)['image']
except ValueError as e:
    print(e)