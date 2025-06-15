#!/usr/bin/env python
"""
handwriting-eval - 手書き文字評価ツール
========================================
新しいモジュール構成でのエントリーポイント
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.eval.cli import main

if __name__ == "__main__":
    sys.exit(main())
