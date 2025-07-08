#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
page_split.py   —   handwriting-sheet slicer v2
------------------------------------------------
1. 背景あり：最大輪郭をページとみなして台形補正
2. 背景なし：補正スキップ
3. Canny→HoughLinesP で左列の 3×3 マス格子を検出しセル切り出し
4. 記入者 No. / 点数枠は固定比率で切り出し（格子基準でもOK）

必要ライブラリ:
    pip install opencv-python numpy
"""

from pathlib import Path
import cv2
import numpy as np
import argparse
import os
import pytesseract

# ----------------------------------------------------------------------
# 0. 汎用ユーティリティ
# ----------------------------------------------------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 1. ページ全体を検出（背景ありの場合）
# ----------------------------------------------------------------------
def find_page_corners(gray):
    """
    最大輪郭をページ外周とみなして 4 隅を返す
    背景が用紙より濃い想定
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 画面が反転してる場合は逆反転
    if bw.mean() < 127:
        bw = cv2.bitwise_not(bw)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("輪郭が取れません")
    page = max(contours, key=cv2.contourArea)

    peri = cv2.arcLength(page, True)
    approx = cv2.approxPolyDP(page, 0.02 * peri, True)
    if len(approx) != 4:
        raise RuntimeError("四角形のページ輪郭が見つかりません")
    return approx.reshape(4, 2).astype("float32")


def order_corners(pts):
    """ 任意順 4 点 → TL, TR, BR, BL (時計回り) """
    pts = pts[np.argsort(pts[:, 0])]  # x で左2 / 右2
    left, right = pts[:2], pts[2:]
    tl = left[np.argmin(left[:, 1])]
    bl = left[np.argmax(left[:, 1])]
    tr = right[np.argmin(right[:, 1])]
    br = right[np.argmax(right[:, 1])]
    return np.array([tl, tr, br, bl], dtype="float32")


def perspective_correct(img, corners, dbg=False):
    """
    corners: 4x2 TL,TR,BR,BL
    出力解像度は実測サイズに合わせて可変
    """
    pts = order_corners(corners)

    wA = np.linalg.norm(pts[2] - pts[3])
    wB = np.linalg.norm(pts[1] - pts[0])
    hA = np.linalg.norm(pts[1] - pts[2])
    hB = np.linalg.norm(pts[0] - pts[3])
    W = int(max(wA, wB))
    H = int(max(hA, hB))

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (W, H))

    if dbg:
        dbg_img = img.copy()
        for (x, y) in pts.astype(int):
            cv2.circle(dbg_img, (x, y), 10, (0, 0, 255), -1)
        cv2.imwrite("dbg_corners.jpg", dbg_img)
        cv2.imwrite("dbg_warped.jpg", warped)
    return warped


# ----------------------------------------------------------------------
# 2. 左列 3×3 マスを内部線検出で抽出
# ----------------------------------------------------------------------
def detect_char_cells(gray, dbg=False):
    """
    罫線を輪郭として抽出し、3 つの「課題マス」を y 順に返す。
    背景が写っていても OK／内部ストローク耐性あり。
    戻り値: [(x1,y1,x2,y2), ...] 3セル
    """
    H, W = gray.shape

    # --- ① 二値化 → 細線を太らせて輪郭を閉じる ---
    th = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 8)
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.dilate(th, kernel, iterations=1)

    # --- ② 輪郭抽出 ---
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    # --- ③ ほぼ正方形・面積閾値でフィルタ ---
    cand = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (H * W) * 0.002 or area > (H * W) * 0.03:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if 0.85 < w / h < 1.15:      # 正方形っぽい
            cand.append((x, y, w, h))

    if len(cand) < 3:
        raise RuntimeError("課題マスらしき四角が 3 つ見つかりません")

    # --- ④ 列ごとにグルーピング → 右列を選択 ---
    # x 座標で 2 クラスタ (空マス列 / 課題列)
    xs = np.array([[c[0]] for c in cand], np.float32)
    _, labels, centers = cv2.kmeans(xs, 2, None,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_PP_CENTERS)
    right_cluster = np.argmax(centers)      # x が大きい方
    right_boxes = [c for c,l in zip(cand, labels.flatten()) if l == right_cluster]

    # y でソートして上→下 3 つを取得
    right_boxes = sorted(right_boxes, key=lambda b: b[1])[:3]
    cells = []
    margin = 10
    for x, y, w, h in right_boxes:
        cells.append((x + margin, y + margin,
                      x + w - margin, y + h - margin))

    if dbg:
        dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x1,y1,x2,y2) in cells:
            cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.imwrite("dbg_cells_contour.jpg", dbg)

    if len(cells) != 3:
        raise RuntimeError("課題マスを 3 つ取得できません")
    return cells

def detect_score_and_comment_boxes(gray, dbg=False):
    """
    右端の評価欄から
        scores : 12 個の点数枠   （正方形寄り）
        cmts   : 12 個のコメント枠（横長）
    を同時に検出して返す

    Returns
    -------
    score_boxes : List[Tuple[x1,y1,x2,y2]]  # len == 12
    cmt_boxes   : List[Tuple[x1,y1,x2,y2]]  # len == 12
    """
    H, W = gray.shape
    roi_x0 = int(W * 0.6)           # 右 40 % だけ見る
    roi    = gray[:, roi_x0:]

    # ---- ① 二値化 & 前処理 ----
    th = cv2.adaptiveThreshold(roi, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 11, 6)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          np.ones((3, 3), np.uint8), 1)
    th = cv2.dilate(th, np.ones((3, 3), np.uint8), 1)

    # ---- ② 輪郭 ----
    cnts, _ = cv2.findContours(th, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)

    score_cand, cmt_cand = [], []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # 小さすぎるもの除外
        if w < 20 or h < 30:  # 20×35
            continue

        ratio = w / h
        area  = w * h
        if area < 600:              # 30×30
            continue

        # (x,y) をフル画像座標系に直す
        X = x + roi_x0

        # --- 点数候補: 正方形〜やや縦長
        if 0.8 < ratio < 1.3:
            score_cand.append((X, y, w, h))
        # --- コメント候補: 横長
        elif ratio > 4.5:
            cmt_cand.append((X, y, w, h))

    if not score_cand or not cmt_cand:
        raise RuntimeError("点数枠 or コメント枠が検出できません")

    if dbg:
        dbg_roi = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        for X,Y,w,h in score_cand:
            cv2.rectangle(dbg_roi, (X - roi_x0, Y), (X - roi_x0 + w, Y + h),
                        (0, 255, 255), 1)
        cv2.imwrite("dbg_score_candidates.jpg", dbg_roi)

    # ---- ③ 右端列だけ残す ----
    def keep_rightmost(boxes, tol=25):
        max_x = max(b[0] for b in boxes)
        return [b for b in boxes if abs(b[0] - max_x) < tol]

    score_cand = keep_rightmost(score_cand, 40)  # 40px 以内
    cmt_cand   = keep_rightmost(cmt_cand)

    # ---- ④ y 昇順に 12 個ずつそろえる ----
    score_cand = sorted(score_cand, key=lambda b: b[1])[:12]
    cmt_cand   = sorted(cmt_cand,   key=lambda b: b[1])[:12]

    # ---- ⑤ margin を内側へ入れて (x1,y1,x2,y2) に変換 ----
    def to_box(lst, margin=4):
        return [(x+margin, y+margin, x+w-margin, y+h-margin)
                for (x,y,w,h) in lst]

    score_boxes = to_box(score_cand)
    cmt_boxes   = to_box(cmt_cand)


    # ---- ⑥ デバッグ描画 ----
    if dbg:
        dbg_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x1,y1,x2,y2 in score_boxes:
            cv2.rectangle(dbg_img,(x1,y1),(x2,y2),(0,0,255),2)   # 赤=点数
        for x1,y1,x2,y2 in cmt_boxes:
            cv2.rectangle(dbg_img,(x1,y1),(x2,y2),(255,0,0),2)   # 青=コメント
        cv2.imwrite("dbg_score_comment_boxes.jpg", dbg_img)

    if len(score_boxes) != 12 or len(cmt_boxes) != 12:
        raise RuntimeError(f"検出数  score:{len(score_boxes)}  cmt:{len(cmt_boxes)}")

    return score_boxes, cmt_boxes

# ----------------------------------------------------------------------
# 3. その他 ROI (固定比率基準、warp 後前提)
# ----------------------------------------------------------------------
def crop_by_ratio(img, ratio):
    H, W = img.shape[:2]
    x, y, w, h = ratio
    x1, y1 = int(W * x), int(H * y)
    x2, y2 = int(W * (x + w)), int(H * (y + h))
    return img[y1:y2, x1:x2]


ROI_FIXED = {
    "writer": (0.73, 0.03, 0.22, 0.05),  # 記入者 No.
    # 右端点数枠 12 個（3×4）
    "scores": [
        (0.96, 0.13 + i*0.06, 0.035, 0.048) for i in range(12)
    ]
}

import cv2, pytesseract, numpy as np

conf_tess = "--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789"

def read_digit(raw):
    """枠付き ROI → 文字を返す (''=失敗)"""
    # ----- 1. 縦罫線除去 -----
    tmp = raw.copy()
    #   a) Canny → HoughLinesP で垂直線を検出
    edges = cv2.Canny(tmp, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=30, minLineLength=raw.shape[0]//2,
                            maxLineGap=10)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            if abs(x1-x2) < 4:          # 垂直
                cv2.line(tmp, (x1,0), (x2,raw.shape[0]-1), 255, 5)

    # ----- 2. 最大輪郭 (数字) を切り抜く -----
    _, bw = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return ""

    # --- A) 面積上位 3 個を並べ替え
    cands = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]

    best = None
    best_score = 0
    for c in cands:
        x,y,w,h = cv2.boundingRect(c)
        patch = bw[y:y+h, x:x+w]
        density = (patch > 0).mean()          # 黒画素率
        sc = cv2.contourArea(c) * density     # 面積×密度
        if sc > best_score:
            best, best_score = (x,y,w,h), sc

    x,y,w,h = best
    roi = raw[y:y+h, x:x+w]           # ← roi 再定義

    # ----- 3. CLAHE ＋ 3 種二値化でリトライ -----
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    roi_eq = clahe.apply(roi)

    for thr in [
        lambda im: cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY,11,2),
        lambda im: cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
        lambda im: cv2.threshold(im,127,255,cv2.THRESH_BINARY)[1]
    ]:
        bin_im = thr(roi_eq)
        up = cv2.resize(bin_im, None, fx=3, fy=3,
                        interpolation=cv2.INTER_CUBIC)
        txt = pytesseract.image_to_string(up, config=conf_tess).strip()
        if txt.isdigit():          # 成功
            return txt

    return ""                      # 3 回とも失敗


# ----------------------------------------------------------------------
# 4. メインパイプライン
# ----------------------------------------------------------------------
def split_page(img_path: Path, out_dir="out", dbg=False):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4-1. ページ補正 (背景有無で分岐)
    try:
        corners = find_page_corners(gray)
        warped = perspective_correct(img, corners, dbg=dbg)
        info = "warp_done"
    except RuntimeError:
        warped = img.copy()
        info = "warp_skipped"

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # 4-2. 文字セル抽出
    try:
        cells = detect_char_cells(warped_gray, dbg=dbg)
    except RuntimeError as e:
        # 左列の大枠矩形 = ページ幅の 0.07 ～ 0.27 を固定比率で切る
        H, W = warped_gray.shape
        W0 = int(W*0.07); W1 = int(W*0.27)
        cell_h = int(H*0.23); gap = int(H*0.31)
        cells = [(W0, int(H*0.11)+i*gap,
                W1, int(H*0.11)+i*gap+cell_h) for i in range(3)]

    char_dir = out_dir / "chars"; ensure_dir(char_dir)
    for i, (x1, y1, x2, y2) in enumerate(cells, 1):
        cv2.imwrite(str(char_dir / f"char_{i}.png"),
                    warped_gray[y1:y2, x1:x2])

    # 4-3. 記入者 No. (固定比率)
    writer_roi = crop_by_ratio(warped_gray, ROI_FIXED["writer"])
    cv2.imwrite(str(out_dir / "writer_id.png"), writer_roi)
    conf = "--psm 10 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(writer_roi, config=conf).strip()
    print("記入者=" + text)

    # 4-4. 点数枠
    score_dir = out_dir / "scores"; ensure_dir(score_dir)
    try:
        score_boxes, cmt_boxes = detect_score_and_comment_boxes(warped_gray, dbg=dbg)
    except RuntimeError as e:
        raise RuntimeError(f"点数枠検出失敗: {e}")
    # 右端の 12 個を切り出し
    score_dir = out_dir / "scores"; ensure_dir(score_dir)
    conf = "--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789"
    for idx, (x1,y1,x2,y2) in enumerate(score_boxes, 1):
        raw = warped_gray[y1:y2, x1:x2]
        digit = read_digit(raw)
        print(f"score_{idx}={digit or '–'}")  # 空文字なら – を表示
        cv2.imwrite(str(score_dir / f"score_{idx}.png"),
                    warped_gray[y1:y2, x1:x2])

    # 4-4. コメント枠
    cmt_dir = out_dir / "comments"; ensure_dir(cmt_dir)
    for idx, (x1,y1,x2,y2) in enumerate(cmt_boxes, 1):
        cv2.imwrite(str(cmt_dir / f"comment_{idx}.png"),
                    warped_gray[y1:y2, x1:x2])

    print(f"[{info}] Crops saved to {out_dir.resolve()}")


# ----------------------------------------------------------------------
# 5. CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("img", type=Path, help="入力画像")
    ap.add_argument("--out_dir", default="out", help="出力ディレクトリ")
    ap.add_argument("--dbg", action="store_true",
                    help="デバッグ画像を保存")
    args = ap.parse_args()
    split_page(args.img, args.out_dir, dbg=args.dbg)

