"""
preprocessing.py
================
画像前処理（台形補正・二値化など）
"""

import cv2
import numpy as np
from pathlib import Path

# ======= Hough 検出パラメータ =======
HOUGH_MINLEN = 70
HOUGH_THRESH = 50
# -----------------------------------

def perspective_correct(img_gray, size=256, dbg=False):
    """
    台形補正を行う
    
    Args:
        img_gray: グレースケール画像
        size: 出力サイズ（正方形）
        dbg: デバッグモード
        
    Returns:
        変換後の画像
        
    Raises:
        ValueError: 枠線が検出できない場合
    """
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


def binarize(img_gray):
    """
    大津の手法で二値化を行う
    
    Args:
        img_gray: グレースケール画像
        
    Returns:
        二値化マスク（0または1）
    """
    _, bw = cv2.threshold(img_gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (bw > 0).astype(np.uint8)


def preprocess(path: Path, size: int, dbg=False):
    """
    画像を読み込み、台形補正・二値化を行う
    
    Args:
        path: 画像ファイルパス
        size: 出力サイズ
        dbg: デバッグモード
        
    Returns:
        tuple: (グレースケール画像, 二値化マスク)
        
    Raises:
        FileNotFoundError: ファイルが見つからない場合
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    try:
        img = perspective_correct(img, size=size, dbg=dbg)
    except ValueError:
        img = cv2.resize(img, (size, size),
                         interpolation=cv2.INTER_AREA)
    return img, binarize(img)


def preprocess_from_array(img_array, size=256, dbg=False):
    """
    配列形式の画像データを前処理する（FastAPI用）
    
    Args:
        img_array: OpenCV画像配列（BGR形式）
        size: 出力サイズ
        dbg: デバッグモード
        
    Returns:
        tuple: (グレースケール画像, 二値化マスク)
    """
    # BGRからグレースケールに変換
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_array
    
    try:
        img_gray = perspective_correct(img_gray, size=size, dbg=dbg)
    except ValueError:
        # 台形補正が失敗した場合は単純リサイズ
        img_gray = cv2.resize(img_gray, (size, size), interpolation=cv2.INTER_AREA)
    
    return img_gray, binarize(img_gray)
