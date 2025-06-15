#!/usr/bin/env python3
"""
test_position_robustness.py
============================
位置補正機能のテスト：同じ形状で位置だけずらした場合のスコア比較
"""

import numpy as np
import cv2
import sys
import os

# パスを追加してsrcモジュールをインポート可能にする
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.eval.metrics import shape_score

def create_circle_mask(size, center, radius):
    """指定位置に円のマスクを作成"""
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    return mask

def create_square_mask(size, center, width):
    """指定位置に正方形のマスクを作成"""
    mask = np.zeros(size, dtype=bool)
    half_w = width // 2
    top = max(0, center[1] - half_w)
    bottom = min(size[0], center[1] + half_w)
    left = max(0, center[0] - half_w)
    right = min(size[1], center[0] + half_w)
    mask[top:bottom, left:right] = True
    return mask

def test_position_robustness():
    """位置補正機能のテスト"""
    print("=== 位置補正機能テスト ===\n")
    
    # テスト設定
    canvas_size = (100, 100)
    shape_size = 20
    
    # 参照形状（中央）
    ref_center = (50, 50)
    ref_circle = create_circle_mask(canvas_size, ref_center, shape_size//2)
    ref_square = create_square_mask(canvas_size, ref_center, shape_size)
    
    # 位置をずらしたテストケース
    test_positions = [
        ("中央", (50, 50)),
        ("右下", (70, 70)),
        ("左上", (30, 30)),
        ("右上", (70, 30)),
        ("左下", (30, 70)),
    ]
    
    print("1. 円形マスクのテスト")
    print("位置\t\tスコア")
    print("-" * 20)
    
    for name, center in test_positions:
        user_circle = create_circle_mask(canvas_size, center, shape_size//2)
        score = shape_score(ref_circle, user_circle)
        print(f"{name}\t\t{score:.3f}")
    
    print("\n2. 正方形マスクのテスト")
    print("位置\t\tスコア")
    print("-" * 20)
    
    for name, center in test_positions:
        user_square = create_square_mask(canvas_size, center, shape_size)
        score = shape_score(ref_square, user_square)
        print(f"{name}\t\t{score:.3f}")
    
    print("\n3. 異なる形状間のテスト（円 vs 正方形）")
    print("位置\t\tスコア")
    print("-" * 20)
    
    for name, center in test_positions:
        user_square = create_square_mask(canvas_size, center, shape_size)
        score = shape_score(ref_circle, user_square)
        print(f"{name}\t\t{score:.3f}")

if __name__ == "__main__":
    test_position_robustness()
