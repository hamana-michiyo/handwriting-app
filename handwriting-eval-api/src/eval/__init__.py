"""
handwriting_eval - 手書き文字評価モジュール
===========================================
"""

from .metrics import shape_score, black_score, white_score, center_score
from .preprocessing import preprocess, perspective_correct, binarize
from .pipeline import evaluate_pair, evaluate_all
from .cli import main

__all__ = [
    "shape_score",
    "black_score", 
    "white_score",
    "center_score",
    "preprocess",
    "perspective_correct",
    "binarize",
    "evaluate_pair",
    "evaluate_all",
    "main"
]
