#!/usr/bin/env python3
"""
test_intensity_analysis.py
==========================
濃淡解析機能のテスト・デモンストレーション
"""

import numpy as np
import cv2
import sys
import os

# パスを追加してsrcモジュールをインポート可能にする
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.eval.metrics import analyze_stroke_intensity, comprehensive_black_score

def create_test_images():
    """テスト用の画像とマスクを生成"""
    size = 100
    
    # 1. 均一な濃度の線（理想的）
    uniform_gray = np.full((size, size), 240, dtype=np.uint8)  # 明るい背景
    cv2.rectangle(uniform_gray, (20, 40), (80, 60), 100, -1)  # 均一な濃度
    uniform_mask = (uniform_gray < 200).astype(np.uint8)
    
    # 2. 濃淡にムラがある線（問題あり）
    uneven_gray = np.full((size, size), 240, dtype=np.uint8)
    # グラデーション効果を追加
    for i in range(20, 81):
        intensity = 100 + int(50 * np.sin((i-20) / 60 * np.pi * 3))  # 濃淡変化
        cv2.line(uneven_gray, (i, 40), (i, 60), intensity, 1)
    uneven_mask = (uneven_gray < 200).astype(np.uint8)
    
    # 3. 薄い線（問題あり）
    thin_gray = np.full((size, size), 240, dtype=np.uint8)
    cv2.rectangle(thin_gray, (20, 40), (80, 60), 200, -1)  # 薄い濃度
    thin_mask = (thin_gray < 220).astype(np.uint8)
    
    return [
        (uniform_gray, uniform_mask, "均一な濃度の線"),
        (uneven_gray, uneven_mask, "濃淡ムラあり"),
        (thin_gray, thin_mask, "薄い線")
    ]

def test_intensity_analysis():
    """濃淡解析機能のテスト"""
    print("=== 濃淡解析機能テスト ===\n")
    
    try:
        test_images = create_test_images()
        
        print("テストケース           平均濃度  濃度標準偏差  濃度CV    均一性スコア")
        print("-" * 75)
        
        for gray, mask, description in test_images:
            analysis = analyze_stroke_intensity(gray, mask)
            
            print(f"{description:<15} {analysis['mean_intensity']:8.1f}  "
                  f"{analysis['intensity_std']:10.1f}  "
                  f"{analysis['intensity_cv']:8.3f}  "
                  f"{analysis['uniformity_score']:10.3f}")
        
        print("\n=== 解釈 ===")
        print("- 平均濃度: 線の濃さ（高いほど濃い）")
        print("- 濃度CV: 濃度のばらつき（低いほど均一）")  
        print("- 均一性スコア: 濃度均一性の総合評価（1.0が理想）")
        
    except Exception as e:
        print(f"テスト中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()

def test_comprehensive_black_score():
    """包括的黒スコア評価のテスト"""
    print("\n\n=== 包括的黒スコア評価テスト ===\n")
    
    test_images = create_test_images()
    ref_gray, ref_mask, ref_desc = test_images[0]  # 基準は均一な線
    
    print("基準: 均一な濃度の線")
    print("比較対象               線幅安定性  濃淡類似度  統合スコア")
    print("-" * 60)
    
    for gray, mask, description in test_images:
        analysis = comprehensive_black_score(ref_gray, ref_mask, gray, mask)
        
        print(f"{description:<15} {analysis['width_stability']:10.3f}  "
              f"{analysis['intensity_similarity']:10.3f}  "
              f"{analysis['total_score']:10.3f}")
    
    print("\n=== 解釈 ===")
    print("- 線幅安定性: 既存の線幅変動係数ベース評価")
    print("- 濃淡類似度: 新しい濃淡均一性ベース評価")
    print("- 統合スコア: 線幅(60%) + 濃淡(40%)の重み付き統合")

def save_test_images():
    """テスト画像を保存（デバッグ用）"""
    test_images = create_test_images()
    
    for i, (gray, mask, description) in enumerate(test_images):
        # グレースケール画像を保存
        cv2.imwrite(f'/workspace/validation/test_intensity_{i}_gray.png', gray)
        # マスクを保存  
        cv2.imwrite(f'/workspace/validation/test_intensity_{i}_mask.png', mask * 255)
        print(f"保存: {description} -> test_intensity_{i}_*.png")

if __name__ == "__main__":
    test_intensity_analysis()
    test_comprehensive_black_score()
    
    # デバッグ用画像保存（オプション）
    print("\n" + "="*50)
    save_test_images()
    print("テスト画像を validation/ フォルダに保存しました")
