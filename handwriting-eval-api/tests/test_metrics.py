"""
test_metrics.py
===============
メトリクス関数のテスト
"""

import pytest
import numpy as np
from src.eval.metrics import (
    shape_score, black_score, white_score, center_score,
    stroke_cv, black_ratio
)


def test_shape_score_identical(sample_mask):
    """同一マスクのIoUは1.0になるべき"""
    score = shape_score(sample_mask, sample_mask)
    assert score == 1.0


def test_shape_score_no_overlap():
    """同じ形状で位置が違う場合は位置補正により1.0になるべき"""
    mask1 = np.zeros((64, 64), dtype=np.uint8)
    mask1[10:20, 10:20] = 1
    
    mask2 = np.zeros((64, 64), dtype=np.uint8)
    mask2[40:50, 40:50] = 1
    
    score = shape_score(mask1, mask2)
    assert score == 1.0  # 位置補正により同じ形状は1.0になる


def test_shape_score_different_shapes():
    """異なる形状のマスクは低いスコアになるべき"""
    # 正方形
    mask1 = np.zeros((64, 64), dtype=np.uint8)
    mask1[20:40, 20:40] = 1
    
    # 細長い長方形
    mask2 = np.zeros((64, 64), dtype=np.uint8)
    mask2[30:34, 10:50] = 1
    
    score = shape_score(mask1, mask2)
    assert 0.0 <= score < 1.0  # 異なる形状は1.0未満


def test_shape_score_empty_masks():
    """空のマスクでのIoUテスト"""
    empty_mask = np.zeros((64, 64), dtype=np.uint8)
    score = shape_score(empty_mask, empty_mask)
    assert score == 0.0


def test_black_ratio(sample_mask):
    """黒画素割合の計算テスト"""
    ratio = black_ratio(sample_mask)
    expected = (20 * 20) / (64 * 64)  # 400/4096
    assert abs(ratio - expected) < 1e-6


def test_black_ratio_full_black():
    """全て黒の場合の黒画素割合"""
    full_mask = np.ones((64, 64), dtype=np.uint8)
    ratio = black_ratio(full_mask)
    assert ratio == 1.0


def test_black_ratio_full_white():
    """全て白の場合の黒画素割合"""
    empty_mask = np.zeros((64, 64), dtype=np.uint8)
    ratio = black_ratio(empty_mask)
    assert ratio == 0.0


def test_center_score_perfect():
    """中央に配置された文字の場スコアは高いはず"""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[30:34, 30:34] = 1  # 中央の4x4
    
    score = center_score(mask)
    assert score > 0.9  # 高スコアであることを確認


def test_center_score_corner():
    """端に配置された文字の場スコアは低いはず"""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[0:4, 0:4] = 1  # 左上の4x4
    
    score = center_score(mask)
    assert score < 0.5  # 低スコアであることを確認


def test_center_score_empty_mask():
    """空のマスクの場スコアは0になるべき"""
    empty_mask = np.zeros((64, 64), dtype=np.uint8)
    score = center_score(empty_mask)
    assert score == 0.0


def test_stroke_cv():
    """線幅変動係数の計算テスト"""
    # 太い線のマスク
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[30:34, 10:50] = 1  # 水平の太い線
    
    cv = stroke_cv(mask)
    assert cv is not None
    assert cv >= 0  # 非負であることを確認


def test_stroke_cv_too_few_points():
    """点が少なすぎる場合はNoneを返すべき"""
    small_mask = np.zeros((64, 64), dtype=np.uint8)
    small_mask[32, 32] = 1  # 1点だけ
    cv = stroke_cv(small_mask)
    assert cv is None


def test_white_score():
    """白スコアの計算テスト"""
    score = white_score(0.2, 0.2)  # 同じ割合
    assert score > 0.9  # 高スコア
    
    score = white_score(0.2, 0.8)  # 大きく異なる
    assert score < 0.1  # 低スコア


def test_black_score():
    """黒スコアの計算テスト"""
    score = black_score(0.1, 0.1)  # 同じCV
    assert score > 0.9  # 高スコア
    
    score = black_score(0.1, 0.5)  # 大きく異なるCV
    assert score < 0.5  # 低スコア
    
    score = black_score(None, 0.1)  # None値の処理
    assert score == 0.0
