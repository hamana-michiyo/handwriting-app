# handwriting-eval

Python スクリプトでお手本画像とユーザ画像を比較し、手書き文字を 4 軸（形・黒・白・場）で評価します。

## 目的

1. ユーザが撮影した手書き文字を  
   - 形（スケール対応ハイブリッド形状評価：位置・サイズ不変）  
   - 黒（線の太さ安定度）  
   - 白（空白バランス）  
   - 場（重心位置）  
   の 4 軸で定量評価
2. Flutter アプリから呼べる HTTP API を最終目標
3. 後段で DL モデルへ置き換え可能なモジュール設計

## 主要機能

### v0.2.4 最新機能（FastAPI導入完了）
- **RESTful API サーバー**: HTTP APIによる評価機能提供
- **Swagger UI**: 自動生成APIドキュメント（/docs）
- **マルチ入力対応**: Base64画像・ファイルアップロード両対応
- **CORS対応**: フロントエンドからの直接アクセス可能

### v0.2.3 機能（Phase 1.5 精密化完了）
- **濃淡解析の精密化**: 局所ムラ検出・エッジ強度評価・筆圧分析の多面的評価
- **線幅評価の改良**: 方向性考慮・サンプリング最適化・ノイズ除去強化
- **包括的強化評価**: 基本・精密・改良機能の選択的統合システム
- **CLIインターフェース拡張**: --enhancedオプションによる精密化機能切り替え

### v0.2.2 機能（Phase 1 完了）
- **パラメータ最適化**: 線幅安定性60% + 濃淡均一性40%の最適バランス確定
- **詳細診断メッセージ**: ユーザー向け改善提案の自動生成
- **計算効率化**: 距離変換キャッシュによる高速処理
- **拡張ロードマップ**: Phase 2-3の詳細計画（筆圧解析、機械学習評価等）

### v0.2.1 機能（線質評価強化）
- **包括的線質評価**: 線幅安定性 + 濃淡均一性の統合評価
- **濃淡解析**: グレースケール画像による筆圧・ムラ検出
- **詳細診断**: 線質問題の具体的な数値化と指摘

### v0.2.0 新機能
- **位置ロバスト形状評価**: 同一形状なら位置に関係なく高スコア（従来1%→100%）
- **スケール対応形状評価**: サイズ違いの相似形を適切に評価（89%+の精度）
- **ハイブリッド評価システム**: 位置補正IoU + 改良Huモーメントの統合
- **マルチスケール探索**: 複数スケールファクターでの最適評価


## Dev Container

このプロジェクトは VSCode の Dev Container で動作します。Docker と Remote - Containers 拡張機能をインストールした上で、以下の手順で開発環境を起動できます。

1. VSCode で本リポジトリを開きます。
2. コマンドパレット（`F1`）を開き、**Remote-Containers: Reopen in Container** を実行します。

Dev Container のビルド後、依存パッケージはすでにインストールされています。コンテナ内のターミナルで以下のコマンドを実行してスクリプトを動かしてください。

```bash
# 新しいモジュール構成での実行
python evaluate.py <reference.jpg> <user.jpg> [-s サイズ] [--dbg] [--json] [--enhanced]

# 例: サンプル画像での評価
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg

# Phase 1.5 精密化機能を使用（濃淡・線幅解析強化）
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg --enhanced

# JSON形式での出力（精密化機能ON）
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg --enhanced --json

# デバッグモード
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg --dbg
```

## FastAPI サーバー使用方法

### サーバー起動
```bash
# APIサーバーを起動
python api_server.py

# または uvicorn直接実行（推奨）
uvicorn api_server:app --reload --host 0.0.0.0 --port 8001
```

### API エンドポイント
- **`GET /`**: API情報
- **`GET /health`**: ヘルスチェック  
- **`POST /evaluate`**: Base64画像評価
- **`POST /evaluate/upload`**: ファイルアップロード評価
- **`GET /docs`**: Swagger UI（自動生成ドキュメント）

### API使用例

#### 1. Base64画像評価
```python
import requests
import base64

# 画像をBase64エンコード
with open("reference.jpg", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode()
with open("user.jpg", "rb") as f:
    user_b64 = base64.b64encode(f.read()).decode()

# API呼び出し
data = {
    "reference_image": ref_b64,
    "target_image": user_b64,
    "size": 256,
    "enhanced_analysis": True  # 精密化機能ON
}

response = requests.post("http://localhost:8001/evaluate", json=data)
result = response.json()

print(f"総合スコア: {result['scores']['total']}%")
print(f"成功: {result['success']}")
```

#### 2. ファイルアップロード評価
```python
import requests

# ファイルアップロード
files = {
    "reference_image": open("reference.jpg", "rb"),
    "target_image": open("user.jpg", "rb")
}
data = {
    "size": 256,
    "enhanced_analysis": True
}

response = requests.post("http://localhost:8001/evaluate/upload", 
                        files=files, data=data)
result = response.json()
```

#### 3. JavaScript/フロントエンド例
```javascript
// ファイルアップロード評価
const formData = new FormData();
formData.append('reference_image', referenceFile);
formData.append('target_image', targetFile);
formData.append('enhanced_analysis', true);

fetch('http://localhost:8001/evaluate/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('評価結果:', data.scores);
});
```

### API レスポンス形式
```json
{
  "scores": {
    "形": 89.5,
    "黒": 92.3,
    "白": 85.7,
    "場": 94.1,
    "total": 90.2,
    "black_details": {
      "analysis_level": "enhanced",
      "enhanced_intensity_similarity": 88.5,
      "improved_width_similarity": 91.2,
      ...
    }
  },
  "success": true,
  "message": "評価が正常に完了しました"
}
```

## 検証・テスト

形状評価システムの性能を検証するためのプログラムが`validation/`フォルダに用意されています：

```bash
# 形状評価手法の比較テスト
python validation/shape_evaluation_comparison.py

# 位置ロバスト性テスト
python validation/test_position_robustness.py

# スケールロバスト性テスト（サイズ違い相似形評価）
python validation/test_scale_robustness.py

# Phase 1 濃淡解析機能テスト
python validation/test_intensity_analysis.py

# Phase 1.5 精密化機能テスト（局所ムラ・方向性・ノイズ除去）
python validation/test_enhanced_analysis.py

# パラメータ最適化ツール
python validation/parameter_optimizer.py
```

## ディレクトリ構成

```
handwriting-eval/
├─ README.md                 # このファイル
├─ requirements.txt          # Python依存関係
├─ evaluate.py               # CLIエントリーポイント
├─ api_server.py             # FastAPI RESTful APIサーバー
├─ test.py                   # 旧スクリプト（互換性のため保持）
├─ src/                      # ソースコード
│   ├─ __init__.py
│   └─ eval/                 # 評価モジュール
│       ├─ __init__.py
│       ├─ preprocessing.py  # 台形補正・二値化など
│       ├─ metrics.py        # 形・黒・白・場 スコア関数（スケール対応ハイブリッド評価）
│       ├─ pipeline.py       # evaluate_pair, evaluate_all など
│       └─ cli.py            # コマンドラインエントリ
├─ tests/                    # テストコード
│   ├─ conftest.py
│   ├─ test_metrics.py
│   ├─ test_pipeline.py
│   ├─ test_preprocessing.py
│   └─ test_integration.py
├─ validation/               # 検証・性能テストプログラム
│   ├─ README.md             # 検証プログラム説明
│   ├─ shape_evaluation_comparison.py    # 形状評価手法比較テスト
│   ├─ test_position_robustness.py       # 位置ロバスト性テスト
│   ├─ test_scale_robustness.py          # スケールロバスト性テスト
│   ├─ test_intensity_analysis.py        # 濃淡解析機能テスト
│   ├─ test_enhanced_analysis.py         # Phase 1.5精密化機能テスト
│   ├─ parameter_optimizer.py            # パラメータ最適化ツール
│   └─ phase1_optimization_results.py    # Phase 1最適化結果
├─ data/                     # データファイル
│   ├─ samples/              # サンプル画像
│   │   ├─ ref_光.jpg        # お手本画像
│   │   ├─ user_光1.jpg      # ユーザー画像
│   │   └─ 光2.jpg           # 追加サンプル
│   └─ README.md             # データ説明
├─ notebooks/                # Jupyter Notebook
│   └─ 01_experiments.ipynb  # 実験・チューニング用
└─ docs/                     # ドキュメント
    ├─ 仕様まとめ.md          # システム仕様
    ├─ 変更履歴.md            # 変更履歴
    └─ スケール対応評価結果.md # v0.2.0テスト結果分析
```