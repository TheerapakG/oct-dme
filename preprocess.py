import cv2
import numpy as np
from pathlib import Path

# TODO: cmd line
# IMG_PATH = Path("./data/dme/Response_f3bc8bc543ac8f076627c71ea8efaf35_L_011.jpg")
IMG_PATH = Path("./data/dme/Non response_0b9cf0f939daa80b6f7c1457ad104140_L_012.jpg")

img = cv2.imread(str(IMG_PATH))
cropped = img[:496, 496:, :]
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
_blur = cv2.bilateralFilter(gray, 32, 150, 150)
blur = cv2.normalize(_blur, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
cv2.imshow("blur", blur)
_, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cs = [c for c in contours if cv2.contourArea(c) > 16 * thresh.shape[0]]
min_p = dict()
max_p = dict()
for c in cs:
    for t in c:
        for p in t:
            if p[0] in min_p:
                min_p[p[0]] = min(p[1], min_p[p[0]])
            else:
                min_p[p[0]] = p[1]
            if p[0] in max_p:
                max_p[p[0]] = max(p[1], max_p[p[0]])
            else:
                max_p[p[0]] = p[1]
new_c = np.array(
    [
        *[[p] for p in sorted(min_p.items())],
        *[[p] for p in sorted(max_p.items(), reverse=True)],
    ]
)
cv2.drawContours(cropped, cs, -1, (0, 0, 255), 3)
cv2.drawContours(cropped, [new_c], -1, (255, 0, 0), 3)
cv2.imshow("selection", cropped)
stencil = np.zeros(gray.shape).astype(gray.dtype)
cv2.fillPoly(stencil, [new_c], (255))
result = cv2.bitwise_and(gray, stencil)
cv2.imshow("result", result)

cv2.waitKey(10000)
