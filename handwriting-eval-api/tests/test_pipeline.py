"""
test_pipeline.py
================
パイプライン機能のテスト
"""

import pytest
import numpy as np
from src.eval.pipeline import evaluate_pair


def test_evaluate_pair(sample_image, sample_mask):
    """ペア評価のテスト"""
    scores = evaluate_pair(sample_image, sample_mask, sample_image, sample_mask)
    
    # 同一画像なので高スコアが期待される
    assert "形" in scores
    assert "黒" in scores
    assert "白" in scores
    assert "場" in scores
    assert "total" in scores
    
    # スコアは0-100の範囲
    for key in ["形", "黒", "白", "場", "total"]:
        assert 0 <= scores[key] <= 100
    
    # 同一画像なので形は100点
    assert scores["形"] == 100.0


def test_evaluate_pair_different():
    """異なる画像のペア評価テスト"""
    # お手本: 中央に正方形
    ref_img = np.full((64, 64), 255, dtype=np.uint8)
    ref_mask = np.zeros((64, 64), dtype=np.uint8)
    ref_mask[20:40, 20:40] = 1
    
    # ユーザー: 少しずれた正方形
    user_img = np.full((64, 64), 255, dtype=np.uint8)
    user_mask = np.zeros((64, 64), dtype=np.uint8)
    user_mask[25:45, 25:45] = 1
    
    scores = evaluate_pair(ref_img, ref_mask, user_img, user_mask)
    
    # 少しずれているので形スコアは100点未満
    assert scores["形"] < 100.0
    assert scores["形"] > 0.0  # でも0点ではない
