import cv2
import os

# 元画像
IMG_PATH = "result/a4_sheet.jpg"
SAVE_DIR = "result/digits"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 読み込み
img = cv2.imread(IMG_PATH)
h, w, _ = img.shape
print(f"画像サイズ: {w}x{h}")
offset_x, offset_y = 30, 50  # 左上の余白
h = h - offset_x  # 下端の余白を除去
w = w - offset_y  # 右端の余白を除去
print(f"画像サイズ: {w}x{h}")
# グリッド設定
rows = 10
cols = 14  # 0〜10 + 3列
cell_width = w // cols  # 右端の余白を考慮
cell_height = h // rows

count = 1

for i in range(rows):
    for j in range(cols):
        x = j * cell_width + offset_x
        y = i * cell_height + offset_y
        roi = img[y:y+cell_height, x:x+cell_width]

        # グレースケール & リサイズ
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))

        # ファイル名にラベル
        label = j if j <= 10 else 10
        filename = f"{label}_{i:02d}_{j:02d}.png"
        cv2.imwrite(os.path.join(SAVE_DIR, filename), resized)
        print(f"保存: {filename}")

cv2.destroyAllWindows()