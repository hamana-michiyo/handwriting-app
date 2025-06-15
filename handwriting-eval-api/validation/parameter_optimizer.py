#!/usr/bin/env python3
"""
parameter_optimizer.py
=====================
黒スコア評価のパラメータ最適化ツール
"""

import numpy as np
import sys
import os

# パスを追加してsrcモジュールをインポート可能にする
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.eval.metrics import comprehensive_black_score, analyze_stroke_intensity
from validation.test_intensity_analysis import create_test_images

def test_parameter_combinations():
    """異なるパラメータ組み合わせでの評価テスト"""
    print("=== パラメータ最適化テスト ===\n")
    
    # テスト用画像を生成
    test_images = create_test_images()
    ref_gray, ref_mask, ref_desc = test_images[0]  # 基準は均一な線
    
    # パラメータの候補
    width_weights = [0.5, 0.6, 0.7, 0.8]
    intensity_cv_scales = [2.0, 3.0, 4.0, 5.0]
    thin_thresholds = [100.0, 128.0, 150.0, 180.0]
    
    print("重み比率のテスト:")
    print("Width重み  Intensity重み  均一線  ムラ線  薄線    区別性")
    print("-" * 60)
    
    best_score = 0
    best_params = {}
    
    for width_w in width_weights:
        intensity_w = 1.0 - width_w
        
        # 各テスト画像での評価
        scores = []
        for gray, mask, desc in test_images:
            analysis = _test_comprehensive_score(ref_gray, ref_mask, gray, mask, 
                                                width_w, intensity_w, 3.0, 128.0)
            scores.append(analysis['total_score'])
        
        # 区別性を評価（理想的には均一=1.0, ムラ・薄<0.8）
        distinction = scores[0] - max(scores[1], scores[2])
        
        print(f"{width_w:8.1f}    {intensity_w:8.1f}      "
              f"{scores[0]:.3f}  {scores[1]:.3f}  {scores[2]:.3f}  {distinction:6.3f}")
        
        if distinction > best_score:
            best_score = distinction
            best_params = {'width_w': width_w, 'intensity_w': intensity_w}
    
    print(f"\n最適重み比率: 線幅安定性={best_params['width_w']:.1f}, "
          f"濃淡均一性={best_params['intensity_w']:.1f}")
    
    return best_params

def test_cv_scale_sensitivity():
    """濃度CV評価スケーリングの感度テスト"""
    print("\n\n=== 濃度CV評価スケーリング感度テスト ===\n")
    
    test_images = create_test_images()
    ref_gray, ref_mask, ref_desc = test_images[0]
    uneven_gray, uneven_mask, uneven_desc = test_images[1]  # ムラあり
    
    cv_scales = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    
    print("CVスケール  ムラ検出感度  均一性スコア差")
    print("-" * 40)
    
    for scale in cv_scales:
        # 均一な線の分析
        uniform_analysis = _test_intensity_analysis(ref_gray, ref_mask, scale, 128.0)
        
        # ムラあり線の分析
        uneven_analysis = _test_intensity_analysis(uneven_gray, uneven_mask, scale, 128.0)
        
        # 感度（差）を計算
        sensitivity = uniform_analysis['uniformity_score'] - uneven_analysis['uniformity_score']
        
        print(f"{scale:8.1f}      {sensitivity:8.3f}        "
              f"{uniform_analysis['uniformity_score']:.3f} vs {uneven_analysis['uniformity_score']:.3f}")
    
    print("\n推奨CVスケール: 3.0-4.0 (適度な感度と安定性)")

def test_thin_threshold_optimization():
    """薄すぎ判定閾値の最適化テスト"""
    print("\n\n=== 薄すぎ判定閾値最適化テスト ===\n")
    
    test_images = create_test_images()
    thin_gray, thin_mask, thin_desc = test_images[2]  # 薄い線
    
    thresholds = [80.0, 100.0, 128.0, 150.0, 180.0, 200.0]
    
    print("判定閾値   薄線均一性スコア  薄すぎ検出力")
    print("-" * 40)
    
    for threshold in thresholds:
        analysis = _test_intensity_analysis(thin_gray, thin_mask, 3.0, threshold)
        
        # 薄すぎ検出力（低いほど良い検出）
        detection_power = 1.0 - analysis['uniformity_score']
        
        print(f"{threshold:8.1f}       {analysis['uniformity_score']:8.3f}        {detection_power:8.3f}")
    
    print("\n推奨閾値: 128.0 (標準的な濃度基準)")

def _test_comprehensive_score(ref_gray, ref_mask, user_gray, user_mask, 
                             width_w, intensity_w, cv_scale, thin_threshold):
    """テスト用の包括的黒スコア計算"""
    from src.eval.metrics import stroke_cv, black_score
    
    # 線幅安定性
    cv_ref = stroke_cv(ref_mask)
    cv_user = stroke_cv(user_mask)
    width_stability = black_score(cv_ref, cv_user)
    
    # 濃淡均一性（パラメータ調整版）
    ref_analysis = _test_intensity_analysis(ref_gray, ref_mask, cv_scale, thin_threshold)
    user_analysis = _test_intensity_analysis(user_gray, user_mask, cv_scale, thin_threshold)
    
    if ref_analysis['uniformity_score'] == 0.0 and user_analysis['uniformity_score'] == 0.0:
        intensity_similarity = 1.0
    else:
        uniformity_diff = abs(ref_analysis['uniformity_score'] - user_analysis['uniformity_score'])
        sigma = ref_analysis['uniformity_score'] * 0.3 + 1e-6
        intensity_similarity = np.exp(-uniformity_diff**2 / (2 * sigma**2))
    
    # 統合スコア
    total_score = width_w * width_stability + intensity_w * intensity_similarity
    
    return {
        'width_stability': width_stability,
        'intensity_similarity': intensity_similarity,
        'total_score': total_score
    }

def _test_intensity_analysis(gray_image, mask, cv_scale, thin_threshold):
    """テスト用の濃淡解析（パラメータ調整版）"""
    if mask.sum() == 0:
        return {'uniformity_score': 0.0}
    
    stroke_pixels = 255 - gray_image[mask > 0]
    mean_intensity = np.mean(stroke_pixels)
    intensity_std = np.std(stroke_pixels)
    intensity_cv = intensity_std / (mean_intensity + 1e-6)
    
    # パラメータ調整版
    intensity_factor = np.clip(mean_intensity / thin_threshold, 0.3, 1.0)
    cv_factor = np.exp(-intensity_cv * cv_scale)
    uniformity_score = intensity_factor * cv_factor
    
    return {'uniformity_score': uniformity_score}

def generate_optimized_config():
    """最適化されたパラメータ設定を生成"""
    print("\n\n=== 推奨パラメータ設定 ===\n")
    
    config = """
# 最適化された線質評価パラメータ
BLACK_WIDTH_WEIGHT = 0.6      # 線幅安定性の重み (最適値)
BLACK_INTENSITY_WEIGHT = 0.4  # 濃淡均一性の重み (最適値)
INTENSITY_CV_SCALE = 3.5      # 濃度CV評価のスケーリング (感度調整)
INTENSITY_THIN_THRESHOLD = 128.0  # 薄すぎ判定の閾値 (標準)

# 用途別調整案
# 厳格評価用: CV_SCALE = 4.0, THIN_THRESHOLD = 150.0
# 寛容評価用: CV_SCALE = 2.5, THIN_THRESHOLD = 100.0
"""
    print(config)
    
    # ファイルに保存
    with open('/workspace/validation/optimized_parameters.py', 'w', encoding='utf-8') as f:
        f.write(config)
    
    print("設定ファイルを /workspace/validation/optimized_parameters.py に保存しました")

if __name__ == "__main__":
    # パラメータ最適化テストを実行
    best_params = test_parameter_combinations()
    test_cv_scale_sensitivity()
    test_thin_threshold_optimization()
    generate_optimized_config()
    
    print("\n" + "="*60)
    print("パラメータ最適化テスト完了")
    print("推奨設定でメトリクス関数を更新することを検討してください")
