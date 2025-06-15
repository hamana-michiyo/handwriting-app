#!/usr/bin/env python3
"""
shape_evaluation_comparison.py
==============================
形状評価手法の比較テスト：IoU vs 位置補正IoU vs ハイブリッド
"""

import numpy as np
import cv2
import sys
import os

# パスを追加してsrcモジュールをインポート可能にする
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.eval.metrics import shape_score, hu_moment_similarity, _calculate_position_corrected_iou

def create_test_shapes():
    """テスト用の形状を作成"""
    canvas_size = (100, 100)
    
    # 1. 円形
    def make_circle(center, radius):
        y, x = np.ogrid[:canvas_size[0], :canvas_size[1]]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        return mask.astype(bool)
    
    # 2. 正方形
    def make_square(center, size):
        mask = np.zeros(canvas_size, dtype=bool)
        half_size = size // 2
        top = max(0, center[1] - half_size)
        bottom = min(canvas_size[0], center[1] + half_size)
        left = max(0, center[0] - half_size)
        right = min(canvas_size[1], center[0] + half_size)
        mask[top:bottom, left:right] = True
        return mask
    
    # 3. 長方形
    def make_rectangle(center, width, height):
        mask = np.zeros(canvas_size, dtype=bool)
        half_w, half_h = width // 2, height // 2
        top = max(0, center[1] - half_h)
        bottom = min(canvas_size[0], center[1] + half_h)
        left = max(0, center[0] - half_w)
        right = min(canvas_size[1], center[0] + half_w)
        mask[top:bottom, left:right] = True
        return mask
    
    return make_circle, make_square, make_rectangle

def original_iou(mask_ref, mask_user):
    """元のIoU計算"""
    union = np.logical_or(mask_ref, mask_user).sum()
    inter = np.logical_and(mask_ref, mask_user).sum()
    if union == 0:
        return 0.0
    return float(inter / union)

def test_shape_evaluation():
    """形状評価手法の比較テスト"""
    print("=== 形状評価手法比較テスト ===\n")
    
    make_circle, make_square, make_rectangle = create_test_shapes()
    
    # テストケース
    test_cases = [
        {
            "name": "同じ円（中央）",
            "ref": make_circle((50, 50), 15),
            "user": make_circle((50, 50), 15),
            "expected": "同一形状・同一位置：全手法で高スコア"
        },
        {
            "name": "同じ円（位置ずれ）",
            "ref": make_circle((50, 50), 15),
            "user": make_circle((70, 70), 15),
            "expected": "同一形状・位置ずれ：位置補正で改善"
        },
        {
            "name": "円 vs 正方形（中央）",
            "ref": make_circle((50, 50), 15),
            "user": make_square((50, 50), 25),
            "expected": "類似形状・同一位置：Huモーメントで差別化"
        },
        {
            "name": "円 vs 正方形（位置ずれ）",
            "ref": make_circle((50, 50), 15),
            "user": make_square((70, 70), 25),
            "expected": "類似形状・位置ずれ：ハイブリッドで最適評価"
        },
        {
            "name": "円 vs 長方形",
            "ref": make_circle((50, 50), 15),
            "user": make_rectangle((50, 50), 40, 10),
            "expected": "異なる形状：全手法で低スコア"
        }
    ]
    
    print(f"{'テストケース':<20} {'元IoU':<8} {'位置補正':<8} {'Hu類似':<8} {'ハイブリッド':<10}")
    print("-" * 70)
    
    for case in test_cases:
        ref_mask = case["ref"]
        user_mask = case["user"]
        
        # 各手法でスコア計算
        original_score = original_iou(ref_mask, user_mask)
        position_corrected = _calculate_position_corrected_iou(ref_mask, user_mask)
        hu_similarity = hu_moment_similarity(ref_mask, user_mask)
        hybrid_score = shape_score(ref_mask, user_mask)
        
        print(f"{case['name']:<20} {original_score:.3f}    {position_corrected:.3f}    {hu_similarity:.3f}    {hybrid_score:.3f}")
    
    print("\n=== 解釈 ===")
    print("- 元IoU: 位置ずれに敏感、純粋な重複度")
    print("- 位置補正: 位置不変、同じ形状なら高スコア")
    print("- Hu類似: 形状記述子ベース、位置・スケール・回転不変")
    print("- ハイブリッド: 位置補正IoU(70%) + Hu類似度(30%)の統合")

def test_weight_sensitivity():
    """重み感度テスト"""
    print("\n\n=== 重み感度テスト ===")
    make_circle, make_square, make_rectangle = create_test_shapes()
    
    ref_mask = make_circle((50, 50), 15)
    user_mask = make_square((70, 70), 25)  # 位置ずれ + 形状違い
    
    iou_score = _calculate_position_corrected_iou(ref_mask, user_mask)
    hu_score = hu_moment_similarity(ref_mask, user_mask)
    
    print(f"位置補正IoU: {iou_score:.3f}")
    print(f"Hu類似度: {hu_score:.3f}")
    print()
    print("重み比率\tハイブリッドスコア")
    print("-" * 25)
    
    for iou_weight in [0.0, 0.3, 0.5, 0.7, 1.0]:
        hu_weight = 1.0 - iou_weight
        hybrid = iou_weight * iou_score + hu_weight * hu_score
        print(f"IoU:{iou_weight:.1f}/Hu:{hu_weight:.1f}\t{hybrid:.3f}")

if __name__ == "__main__":
    test_shape_evaluation()
    test_weight_sensitivity()
