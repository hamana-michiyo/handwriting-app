"""
conftest.py
===========
pytest の共通設定とフィクスチャ
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_mask():
    """テスト用のサンプルマスク"""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 1  # 中央に20x20の正方形
    return mask


@pytest.fixture
def sample_image():
    """テスト用のサンプル画像"""
    img = np.full((64, 64), 255, dtype=np.uint8)  # 白背景
    img[20:40, 20:40] = 0  # 中央に20x20の黒い正方形
    return img


@pytest.fixture
def test_data_dir():
    """テストデータディレクトリ"""
    return Path(__file__).parent.parent / "data"
