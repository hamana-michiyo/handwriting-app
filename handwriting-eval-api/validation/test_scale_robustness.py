#!/usr/bin/env python3
"""
スケールロバスト性テスト - サイズ違いの相似形評価を検証する
"""

import numpy as np
import cv2
import sys
import os

# パスを追加してsrcモジュールをインポート可能にする
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.eval.metrics import shape_score, _calculate_scale_corrected_iou, _improved_hu_moment_similarity

def create_circle_mask(size, radius, center=None):
    """円のマスクを作成"""
    if center is None:
        center = (size // 2, size // 2)
    
    y, x = np.ogrid[:size, :size]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    return mask.astype(bool)

def create_square_mask(size, side_length, center=None):
    """正方形のマスクを作成"""
    if center is None:
        center = (size // 2, size // 2)
    
    mask = np.zeros((size, size), dtype=bool)
    half_side = side_length // 2
    
    left = max(0, center[0] - half_side)
    right = min(size, center[0] + half_side)
    top = max(0, center[1] - half_side)
    bottom = min(size, center[1] + half_side)
    
    mask[top:bottom, left:right] = True
    return mask

def test_scale_robustness():
    """スケールロバスト性テスト"""
    print("=== スケールロバスト性テスト ===\n")
    
    canvas_size = 100
    ref_radius = 15
    
    # 基準となる円（中央）
    ref_mask = create_circle_mask(canvas_size, ref_radius)
    print(f"基準マスクサイズ: {ref_mask.shape}, 有効ピクセル数: {ref_mask.sum()}")
    
    print("基準: 半径15の円（中央）")
    print(f"テストケース               スケール補正IoU   改良Hu類似度   ハイブリッドスコア")
    print("-" * 80)
    
    # 同じサイズの円（中央）
    same_circle = create_circle_mask(canvas_size, ref_radius)
    scale_iou = _calculate_scale_corrected_iou(ref_mask, same_circle)
    hu_sim = _improved_hu_moment_similarity(ref_mask, same_circle)
    hybrid = shape_score(ref_mask, same_circle)
    print(f"同じ円（r=15, 中央）          {scale_iou:.3f}         {hu_sim:.3f}         {hybrid:.3f}")
    
    # 位置ずれした同じサイズの円
    shifted_circle = create_circle_mask(canvas_size, ref_radius, center=(35, 35))
    scale_iou = _calculate_scale_corrected_iou(ref_mask, shifted_circle)
    hu_sim = _improved_hu_moment_similarity(ref_mask, shifted_circle)
    hybrid = shape_score(ref_mask, shifted_circle)
    print(f"同じ円（r=15, 位置ずれ）       {scale_iou:.3f}         {hu_sim:.3f}         {hybrid:.3f}")
    
    # 小さい円（半径10）
    small_circle = create_circle_mask(canvas_size, 10)
    scale_iou = _calculate_scale_corrected_iou(ref_mask, small_circle)
    hu_sim = _improved_hu_moment_similarity(ref_mask, small_circle)
    hybrid = shape_score(ref_mask, small_circle)
    print(f"小さい円（r=10, 中央）         {scale_iou:.3f}         {hu_sim:.3f}         {hybrid:.3f}")
    
    # 大きい円（半径20）
    large_circle = create_circle_mask(canvas_size, 20)
    scale_iou = _calculate_scale_corrected_iou(ref_mask, large_circle)
    hu_sim = _improved_hu_moment_similarity(ref_mask, large_circle)
    hybrid = shape_score(ref_mask, large_circle)
    print(f"大きい円（r=20, 中央）         {scale_iou:.3f}         {hu_sim:.3f}         {hybrid:.3f}")
    
    # 2倍サイズの円（半径30）
    double_circle = create_circle_mask(canvas_size, 30)
    scale_iou = _calculate_scale_corrected_iou(ref_mask, double_circle)
    hu_sim = _improved_hu_moment_similarity(ref_mask, double_circle)
    hybrid = shape_score(ref_mask, double_circle)
    print(f"2倍円（r=30, 中央）           {scale_iou:.3f}         {hu_sim:.3f}         {hybrid:.3f}")
    
    # 位置ずれ + スケール違い
    shifted_large = create_circle_mask(canvas_size, 20, center=(35, 35))
    scale_iou = _calculate_scale_corrected_iou(ref_mask, shifted_large)
    hu_sim = _improved_hu_moment_similarity(ref_mask, shifted_large)
    hybrid = shape_score(ref_mask, shifted_large)
    print(f"大きい円（r=20, 位置ずれ）      {scale_iou:.3f}         {hu_sim:.3f}         {hybrid:.3f}")
    
    print("\n=== 異なる形状のテスト ===")
    
    # 正方形（一辺20）
    square = create_square_mask(canvas_size, 20)
    scale_iou = _calculate_scale_corrected_iou(ref_mask, square)
    hu_sim = _improved_hu_moment_similarity(ref_mask, square)
    hybrid = shape_score(ref_mask, square)
    print(f"正方形（一辺20, 中央）         {scale_iou:.3f}         {hu_sim:.3f}         {hybrid:.3f}")
    
    # 小さい正方形（一辺15）
    small_square = create_square_mask(canvas_size, 15)
    scale_iou = _calculate_scale_corrected_iou(ref_mask, small_square)
    hu_sim = _improved_hu_moment_similarity(ref_mask, small_square)
    hybrid = shape_score(ref_mask, small_square)
    print(f"小さい正方形（一辺15, 中央）    {scale_iou:.3f}         {hu_sim:.3f}         {hybrid:.3f}")
    
    print("\n=== 解釈 ===")
    print("- スケール補正IoU: 最適なスケールでの位置補正IoU")
    print("- 改良Hu類似度: matchShapes + 基本形状記述子")
    print("- ハイブリッドスコア: IoU(70%) + Hu(30%)の統合")
    print("\n期待される結果:")
    print("- 同じ形状（円）のスケール違いは高いスコア（0.8+）")
    print("- 異なる形状は適度なスコア（形状による）")
    print("- 位置ずれは影響を受けない")

if __name__ == "__main__":
    test_scale_robustness()
