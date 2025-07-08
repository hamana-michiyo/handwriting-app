import cv2
import numpy as np

# 画像読み込み
img = cv2.imread("../result/chars/char_1.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("画像が見つかりません。パスを確認してください。")

# ガウシアン
blur = cv2.GaussianBlur(img, (3, 3), 0)

# 二値化
_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

# モルフォロジー（ちょい弱め）
kernel = np.ones((2, 2), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# 反転して背景白・文字黒
result = cv2.bitwise_not(opening)

# 必要ならカラーに戻す
result_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

cv2.imwrite("result/chars/char_1_gaussian_removed.png", result)

