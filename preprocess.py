import cv2
import numpy as np
from pathlib import Path

# TODO: cmd line
IMG_PATH = Path("./data/dme/Non response_0b9cf0f939daa80b6f7c1457ad104140_L_012.jpg")

img = cv2.imread(str(IMG_PATH))
cropped = img[:496, 496:, :]
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(gray, 32, 75, 75)
cv2.imshow("blur", blur)
region = (blur > 47).astype(np.uint8) * 255
cv2.imshow("region", region)
contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c = max(contours, key=cv2.contourArea)
cv2.drawContours(cropped, [c], -1, (0, 0, 255), 3)
cv2.imshow("selection", cropped)
stencil = np.zeros(gray.shape).astype(gray.dtype)
cv2.fillPoly(stencil, [c], (255))
result = cv2.bitwise_and(gray, stencil)
cv2.imshow("result", result)

cv2.waitKey(0)
