"""
handwriting_eval_all.py
=======================
お手本画像とユーザ画像を比較し、4 軸（形・黒・白・場）を 0–100 点で評価
-------------------------------------------------------------
pip install opencv-python numpy

使い方:
    python handwriting_eval_all.py ref.jpg user.jpg [-s 256] [--dbg]

出力:
    {
      "形":  93.2,
      "黒":  88.4,
      "白":  82.1,
      "場":  93.0,
      "total": 89.7
    }
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

# ======= 重みパラメータ (0–1 の合計=1) =======
SHAPE_W  = 0.30
BLACK_W  = 0.20
WHITE_W  = 0.30
CENTER_W = 0.20
# -------------------------------------------

# ======= Hough 検出パラメータ =======
HOUGH_MINLEN = 70
HOUGH_THRESH = 50
# -----------------------------------

# ------------------------------------------------------------------
# 台形補正
# ------------------------------------------------------------------
def perspective_correct(img_gray, size=256, dbg=False):
    blur  = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=HOUGH_THRESH,
                            minLineLength=HOUGH_MINLEN,
                            maxLineGap=10)
    if lines is None:
        raise ValueError("枠線が検出できません")

    h_lines, v_lines = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(y2 - y1) < abs(x2 - x1) * 0.2:      # 水平
            h_lines.append((y1 + y2) / 2)
        elif abs(x2 - x1) < abs(y2 - y1) * 0.2:    # 垂直
            v_lines.append((x1 + x2) / 2)

    if len(h_lines) < 2 or len(v_lines) < 2:
        raise ValueError("水平/垂直線が不足")

    quad = np.array([[min(v_lines), min(h_lines)],
                     [max(v_lines), min(h_lines)],
                     [max(v_lines), max(h_lines)],
                     [min(v_lines), max(h_lines)]],
                    dtype="float32")
    dst  = np.array([[0, 0],
                     [size-1, 0],
                     [size-1, size-1],
                     [0, size-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img_gray, M, (size, size))

    if dbg:
        cv2.imshow("edges", edges)
        cv2.imshow("warped", warped)
        cv2.waitKey(0); cv2.destroyAllWindows()

    return warped

# ------------------------------------------------------------------
# 基本ユーティリティ
# ------------------------------------------------------------------
def binarize(img_gray):
    _, bw = cv2.threshold(img_gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (bw > 0).astype(np.uint8)

def preprocess(path: Path, size: int, dbg=False):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    try:
        img = perspective_correct(img, size=size, dbg=dbg)
    except ValueError:
        img = cv2.resize(img, (size, size),
                         interpolation=cv2.INTER_AREA)
    return img, binarize(img)

# ------------------------------------------------------------------
# 形（IoU）
# ------------------------------------------------------------------
def shape_score(mask_ref, mask_user):
    union  = np.logical_or(mask_ref, mask_user).sum()
    inter  = np.logical_and(mask_ref, mask_user).sum()
    if union == 0:  # 万一空
        return 0.0
    iou = inter / union      # 0–1
    return iou               # 高いほど◎

# ------------------------------------------------------------------
# 黒（線幅ばらつき差）
# ------------------------------------------------------------------
def stroke_cv(mask):
    # 距離変換で各点の半径を推定 -> 線幅=半径*2
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    widths = dist[mask > 0] * 2
    widths = widths[widths > 0.5]          # 端の0を除外
    if len(widths) < 20:                   # 点が少なすぎ
        return None
    return widths.std() / widths.mean()    # 変動係数 (CV)

def black_score(cv_ref, cv_user):
    if cv_ref is None or cv_user is None:
        return 0.0
    sigma = cv_ref * 0.5 + 1e-6
    return np.exp(-((cv_user - cv_ref) ** 2) / (2 * sigma ** 2))

# ------------------------------------------------------------------
# 白（黒画素割合差） & 場（重心）
# ------------------------------------------------------------------
def black_ratio(mask):
    return mask.mean()   # 0–1

def white_score(r_ref, r_user):
    sigma = r_ref * 0.5 + 1e-6
    return np.exp(-((r_user - r_ref) ** 2) / (2 * sigma ** 2))

def center_score(mask):
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    cy, cx = ys.mean(), xs.mean()
    dist = np.hypot((cx - w/2)/(w/2),
                    (cy - h/2)/(h/2))
    return 1.0 - min(dist, 1.0)

# ------------------------------------------------------------------
# メイン評価
# ------------------------------------------------------------------
def evaluate(ref_img, ref_mask, user_img, user_mask):
    # 形
    s_score = shape_score(ref_mask, user_mask)

    # 黒
    cv_ref  = stroke_cv(ref_mask)
    cv_user = stroke_cv(user_mask)
    b_score = black_score(cv_ref, cv_user)

    # 白
    r_ref  = black_ratio(ref_mask)
    r_user = black_ratio(user_mask)
    w_score = white_score(r_ref, r_user)

    # 場
    c_score = center_score(user_mask)

    # 総合
    total = (SHAPE_W  * s_score +
             BLACK_W  * b_score +
             WHITE_W  * w_score +
             CENTER_W * c_score)

    return {
        "形":   round(s_score * 100, 1),
        "黒":   round(b_score * 100, 1),
        "白":   round(w_score * 100, 1),
        "場":   round(c_score * 100, 1),
        "total": round(total * 100, 1)
    }

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path, help="お手本画像")
    ap.add_argument("target",    type=Path, help="判定対象画像")
    ap.add_argument("-s", "--size", type=int, default=256,
                    help="正方リサイズpx")
    ap.add_argument("--dbg", action="store_true",
                    help="デバッグ表示オン")
    args = ap.parse_args()

    ref_img,  ref_mask  = preprocess(args.reference, args.size, args.dbg)
    user_img, user_mask = preprocess(args.target,    args.size, args.dbg)

    scores = evaluate(ref_img, ref_mask, user_img, user_mask)
    print(scores)
