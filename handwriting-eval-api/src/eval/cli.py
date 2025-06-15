"""
cli.py
======
コマンドラインインターフェース
"""

import argparse
import json
from pathlib import Path
from .preprocessing import preprocess
from .pipeline import evaluate_pair


def main():
    """
    メインのコマンドラインエントリーポイント
    """
    ap = argparse.ArgumentParser(
        description="手書き文字評価ツール：お手本とユーザー画像を4軸で比較評価"
    )
    ap.add_argument("reference", type=Path, help="お手本画像")
    ap.add_argument("target",    type=Path, help="判定対象画像")
    ap.add_argument("-s", "--size", type=int, default=256,
                    help="正方リサイズpx（デフォルト: 256）")
    ap.add_argument("--dbg", action="store_true",
                    help="デバッグ表示オン")
    ap.add_argument("--enhanced", action="store_true",
                    help="Phase 1.5精密化機能を使用（濃淡・線幅解析を強化）")
    ap.add_argument("--json", action="store_true",
                    help="JSON形式で出力")
    args = ap.parse_args()

    try:
        ref_img,  ref_mask  = preprocess(args.reference, args.size, args.dbg)
        user_img, user_mask = preprocess(args.target,    args.size, args.dbg)

        scores = evaluate_pair(ref_img, ref_mask, user_img, user_mask, args.enhanced)
        
        if args.json:
            print(json.dumps(scores, ensure_ascii=False, indent=2))
        else:
            print(scores)
            
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません - {e}")
        return 1
    except ValueError as e:
        print(f"エラー: {e}")
        return 1
    except Exception as e:
        print(f"予期しないエラー: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
