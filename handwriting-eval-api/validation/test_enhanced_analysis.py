#!/usr/bin/env python3
"""
test_enhanced_analysis.py
========================
Phase 1.5 精密化機能のテスト・デモンストレーション
"""

import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt

# パスを追加してsrcモジュールをインポート可能にする
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.eval.metrics import (
    enhanced_intensity_analysis, 
    improved_width_analysis,
    comprehensive_enhanced_black_score,
    analyze_stroke_intensity,
    stroke_cv
)
from validation.test_intensity_analysis import create_test_images

def create_enhanced_test_images():
    """精密化機能テスト用の特殊画像を生成"""
    test_images = []
    
    # 1. 局所ムラありの線
    img1 = np.full((100, 100), 255, dtype=np.uint8)
    # 濃度が局所的に変化する線
    for i in range(30, 70):
        for j in range(20, 80):
            if 35 <= i <= 40:  # 中央の線
                if j < 40:
                    intensity = 50 + np.random.normal(0, 10)  # 濃い部分
                elif j < 60:
                    intensity = 120 + np.random.normal(0, 15)  # 薄い部分
                else:
                    intensity = 80 + np.random.normal(0, 8)   # 中間部分
                img1[i, j] = np.clip(255 - intensity, 0, 255)
    
    mask1 = (img1 < 200).astype(np.uint8) * 255
    test_images.append((img1, mask1, "局所ムラあり"))
    
    # 2. 方向性のある複雑な線（縦画・横画組み合わせ）
    img2 = np.full((100, 100), 255, dtype=np.uint8)
    # 縦画（太め）
    img2[20:80, 35:45] = 100
    # 横画（細め）
    img2[45:50, 25:75] = 120
    
    mask2 = (img2 < 200).astype(np.uint8) * 255
    test_images.append((img2, mask2, "縦横画混合"))
    
    # 3. ノイズの多い線
    img3 = np.full((100, 100), 255, dtype=np.uint8)
    # メインの線
    img3[35:40, 20:80] = 80
    # 小さなノイズ点を追加
    for _ in range(10):
        y, x = np.random.randint(10, 90, 2)
        img3[y:y+2, x:x+2] = 150
    
    mask3 = (img3 < 200).astype(np.uint8) * 255
    test_images.append((img3, mask3, "ノイズ混入"))
    
    return test_images

def test_enhanced_intensity_analysis():
    """精密濃淡解析のテスト"""
    print("=== 精密濃淡解析テスト (Phase 1.5) ===\n")
    
    # 基本テスト画像
    basic_images = create_test_images()
    # 精密化テスト画像
    enhanced_images = create_enhanced_test_images()
    
    all_images = basic_images + enhanced_images
    
    print("テストケース           基本均一性  精密均一性  局所均一性  境界明瞭度  筆圧一貫性")
    print("-" * 85)
    
    for gray, mask, description in all_images:
        # 基本解析
        basic_result = analyze_stroke_intensity(gray, mask)
        
        # 精密解析
        enhanced_result = enhanced_intensity_analysis(gray, mask)
        
        print(f"{description:<15} {basic_result['uniformity_score']:10.3f}  "
              f"{enhanced_result['enhanced_uniformity_score']:10.3f}  "
              f"{enhanced_result['local_analysis']['local_uniformity']:10.3f}  "
              f"{enhanced_result['gradient_analysis']['boundary_clarity']:10.3f}  "
              f"{enhanced_result['pressure_analysis']['pressure_consistency']:10.3f}")
    
    print("\n=== 精密化効果の評価 ===")
    print("- 精密均一性: 複数要素を統合した総合評価")
    print("- 局所均一性: スライディングウィンドウによる局所ムラ検出")
    print("- 境界明瞭度: Sobelフィルタによるエッジ強度評価")
    print("- 筆圧一貫性: ヒストグラム分析と複数閾値評価")

def test_improved_width_analysis():
    """改良線幅解析のテスト"""
    print("\n\n=== 改良線幅解析テスト (Phase 1.5) ===\n")
    
    # 精密化テスト画像を使用
    enhanced_images = create_enhanced_test_images()
    
    print("テストケース           基本CV     改良CV     方向一貫性  サンプリング改善  ノイズ除去効果")
    print("-" * 90)
    
    for gray, mask, description in enhanced_images:
        # 基本解析
        basic_cv = stroke_cv(mask)
        
        # 改良解析
        improved_result = improved_width_analysis(mask)
        
        if basic_cv is not None and improved_result['improved_cv'] is not None:
            direction_consistency = improved_result['directional_analysis']['direction_consistency']
            sampling_improvement = improved_result['sampling_analysis']['sampling_improvement']
            noise_improvement = improved_result['noise_analysis']['max_improvement']
            
            print(f"{description:<15} {basic_cv:10.3f}  "
                  f"{improved_result['improved_cv']:9.3f}  "
                  f"{direction_consistency:10.3f}  "
                  f"{sampling_improvement:15.3f}  "
                  f"{noise_improvement:13.3f}")
        else:
            print(f"{description:<15} {'N/A':<10}  {'N/A':<9}  {'N/A':<10}  {'N/A':<15}  {'N/A':<13}")
    
    print("\n=== 改良効果の評価 ===")
    print("- 改良CV: 方向性・サンプリング・ノイズ除去を考慮した精密CV")
    print("- 方向一貫性: 縦画・横画・斜線の方向別評価の一貫性")
    print("- サンプリング改善: 等間隔・適応的サンプリングによる改善度")
    print("- ノイズ除去効果: 端点・小ノイズ・外れ値除去による改善度")

def test_comprehensive_enhanced_score():
    """包括的強化黒スコア評価のテスト"""
    print("\n\n=== 包括的強化黒スコア評価テスト (Phase 1.5) ===\n")
    
    # テスト画像
    all_images = create_test_images() + create_enhanced_test_images()
    ref_gray, ref_mask, ref_desc = all_images[0]  # 基準は均一な線
    
    print("比較対象               基本スコア  強化スコア  精密濃淡類似度  改良線幅類似度")
    print("-" * 80)
    
    for gray, mask, description in all_images:
        # 基本評価
        basic_result = comprehensive_enhanced_black_score(
            ref_gray, ref_mask, gray, mask, 
            use_enhanced=False, use_improved_width=False
        )
        
        # 強化評価
        enhanced_result = comprehensive_enhanced_black_score(
            ref_gray, ref_mask, gray, mask, 
            use_enhanced=True, use_improved_width=True
        )
        
        basic_score = basic_result['total_score']
        enhanced_score = enhanced_result.get('enhanced_total_score', basic_score)
        enhanced_intensity_sim = enhanced_result.get('enhanced_intensity_similarity', 0.0)
        improved_width_sim = enhanced_result.get('improved_width_similarity', 0.0)
        
        print(f"{description:<15} {basic_score:10.3f}  "
              f"{enhanced_score:10.3f}  "
              f"{enhanced_intensity_sim:14.3f}  "
              f"{improved_width_sim:14.3f}")
    
    print("\n=== Phase 1.5 強化機能の効果 ===")
    print("- 基本スコア: Phase 1の線幅安定性60% + 濃淡均一性40%")
    print("- 強化スコア: 精密化機能を統合した包括的評価")
    print("- 精密濃淡類似度: 局所・勾配・筆圧を含む多面的濃淡評価")
    print("- 改良線幅類似度: 方向性・サンプリング・ノイズ除去を含む精密線幅評価")

def save_enhanced_test_images():
    """精密化テスト画像の保存"""
    enhanced_images = create_enhanced_test_images()
    
    for i, (gray, mask, description) in enumerate(enhanced_images):
        # グレースケール画像を保存
        cv2.imwrite(f'/workspace/validation/test_enhanced_{i}_gray.png', gray)
        # マスク画像を保存
        cv2.imwrite(f'/workspace/validation/test_enhanced_{i}_mask.png', mask)
        print(f"保存: test_enhanced_{i}_*.png ({description})")

def demonstrate_phase15_improvements():
    """Phase 1.5 改善効果のデモンストレーション"""
    print("\n\n" + "="*60)
    print("Phase 1.5 精密化機能 デモンストレーション")
    print("="*60)
    
    print("\n【改善項目】")
    print("A. 濃淡解析の精密化:")
    print("   1. 局所的濃度分析 - スライディングウィンドウによる局所ムラ検出")
    print("   2. 濃度勾配解析 - Sobelフィルタによるエッジ強度評価")
    print("   3. 筆圧推定精度向上 - 複数閾値・ヒストグラム分析")
    
    print("\nB. 線幅評価の改良:")
    print("   1. 方向性を考慮した線幅測定 - 縦画・横画・斜線の方向別解析")
    print("   2. サンプリング密度の最適化 - 等間隔・適応的サンプリング")
    print("   3. ノイズ除去の強化 - 端点効果・小ノイズ・外れ値の除去")
    
    print("\n【期待効果】")
    print("- より精密な線質評価（局所的な問題点の検出）")
    print("- ノイズに対するロバスト性向上")
    print("- 方向性を考慮した適切な線幅評価")
    print("- 筆圧変化の詳細な解析")
    
    print("\n【使用方法】")
    print("# 精密化機能ON")
    print("result = comprehensive_enhanced_black_score(")
    print("    ref_gray, ref_mask, user_gray, user_mask,")
    print("    use_enhanced=True, use_improved_width=True)")
    
    print("\n# 後方互換性（精密化機能OFF）")
    print("result = comprehensive_enhanced_black_score(")
    print("    ref_gray, ref_mask, user_gray, user_mask,")
    print("    use_enhanced=False, use_improved_width=False)")

if __name__ == "__main__":
    # Phase 1.5 精密化機能のテストを実行
    test_enhanced_intensity_analysis()
    test_improved_width_analysis()
    test_comprehensive_enhanced_score()
    
    # テスト画像の保存
    save_enhanced_test_images()
    
    # デモンストレーション
    demonstrate_phase15_improvements()
    
    print("\n" + "="*60)
    print("Phase 1.5 精密化機能テスト完了")
    print("黒スコア評価がより精密で詳細な線質分析に進化しました")
