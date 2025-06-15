"""
pipeline.py
===========
評価パイプライン（evaluate_pair, evaluate_all など）
"""

import numpy as np
from .metrics import (
    shape_score, black_score, white_score, center_score, stroke_cv, black_ratio, 
    comprehensive_black_score, comprehensive_enhanced_black_score
)
from .preprocessing import preprocess

# ======= 重みパラメータ (0–1 の合計=1) =======
SHAPE_W  = 0.30
BLACK_W  = 0.20
WHITE_W  = 0.30
CENTER_W = 0.20
# -------------------------------------------

def evaluate_pair(ref_img, ref_mask, user_img, user_mask, enhanced_analysis=False):
    """
    お手本とユーザー画像のペアを評価する
    
    Args:
        ref_img: お手本のグレースケール画像
        ref_mask: お手本の二値化マスク
        user_img: ユーザーのグレースケール画像  
        user_mask: ユーザーの二値化マスク
        enhanced_analysis: Phase 1.5精密化機能を使用するか
        
    Returns:
        dict: 各軸のスコアと総合スコア
    """
    # 形
    s_score = shape_score(ref_mask, user_mask)

    # 黒（包括的評価：線幅安定性 + 濃淡均一性）
    if enhanced_analysis:
        # Phase 1.5 精密化機能
        black_analysis = comprehensive_enhanced_black_score(
            ref_img, ref_mask, user_img, user_mask,
            use_enhanced=True, use_improved_width=True
        )
        b_score = black_analysis.get('enhanced_total_score', black_analysis['total_score'])
    else:
        # 標準機能（後方互換性）
        black_analysis = comprehensive_black_score(ref_img, ref_mask, user_img, user_mask)
        b_score = black_analysis['total_score']

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

    # 詳細情報の構築
    black_details = {
        "width_stability": round(float(black_analysis['width_stability'] * 100), 1),
        "intensity_similarity": round(float(black_analysis['intensity_similarity'] * 100), 1),
        "ref_intensity_cv": round(float(black_analysis['ref_analysis']['intensity_cv']), 3),
        "user_intensity_cv": round(float(black_analysis['user_analysis']['intensity_cv']), 3)
    }
    
    # Phase 1.5 精密化機能の詳細情報追加
    if enhanced_analysis and 'enhanced_total_score' in black_analysis:
        black_details.update({
            "enhanced_total_score": round(float(black_analysis['enhanced_total_score'] * 100), 1),
            "enhanced_intensity_similarity": round(float(black_analysis.get('enhanced_intensity_similarity', 0) * 100), 1),
            "improved_width_similarity": round(float(black_analysis.get('improved_width_similarity', 0) * 100), 1),
            "analysis_level": "enhanced"
        })
        
        # 精密濃淡解析の詳細
        if 'ref_enhanced_intensity' in black_analysis:
            ref_enhanced = black_analysis['ref_enhanced_intensity']
            user_enhanced = black_analysis['user_enhanced_intensity']
            black_details["enhanced_details"] = {
                "local_uniformity_ref": round(float(ref_enhanced['local_analysis']['local_uniformity']), 3),
                "local_uniformity_user": round(float(user_enhanced['local_analysis']['local_uniformity']), 3),
                "boundary_clarity_ref": round(float(ref_enhanced['gradient_analysis']['boundary_clarity']), 3),
                "boundary_clarity_user": round(float(user_enhanced['gradient_analysis']['boundary_clarity']), 3),
                "pressure_consistency_ref": round(float(ref_enhanced['pressure_analysis']['pressure_consistency']), 3),
                "pressure_consistency_user": round(float(user_enhanced['pressure_analysis']['pressure_consistency']), 3)
            }
        
        # 改良線幅解析の詳細
        if 'ref_improved_width' in black_analysis:
            ref_width = black_analysis['ref_improved_width']
            user_width = black_analysis['user_improved_width']
            if ref_width['improved_cv'] is not None and user_width['improved_cv'] is not None:
                black_details["width_details"] = {
                    "improved_cv_ref": round(float(ref_width['improved_cv']), 3),
                    "improved_cv_user": round(float(user_width['improved_cv']), 3),
                    "direction_consistency_ref": round(float(ref_width['directional_analysis']['direction_consistency']), 3),
                    "direction_consistency_user": round(float(user_width['directional_analysis']['direction_consistency']), 3)
                }
    else:
        black_details["analysis_level"] = "standard"

    return {
        "形":   round(float(s_score * 100), 1),
        "黒":   round(float(b_score * 100), 1),
        "白":   round(float(w_score * 100), 1),
        "場":   round(float(c_score * 100), 1),
        "total": round(float(total * 100), 1),
        "black_details": black_details
    }


def evaluate_all(ref_path, user_paths, size=256, dbg=False, enhanced_analysis=False):
    """
    お手本に対して複数のユーザー画像を評価する
    
    Args:
        ref_path: お手本画像のパス
        user_paths: ユーザー画像のパスのリスト
        size: 画像サイズ
        dbg: デバッグモード
        enhanced_analysis: Phase 1.5精密化機能を使用するか
        
    Returns:
        list: 各ユーザー画像の評価結果
    """
    ref_img, ref_mask = preprocess(ref_path, size, dbg)
    
    results = []
    for user_path in user_paths:
        try:
            user_img, user_mask = preprocess(user_path, size, dbg)
            scores = evaluate_pair(ref_img, ref_mask, user_img, user_mask, enhanced_analysis)
            results.append({
                "file": str(user_path),
                "scores": scores
            })
        except Exception as e:
            results.append({
                "file": str(user_path),
                "error": str(e)
            })
    
    return results
