# 🖋️ 美文字評価システム（Bimoji Evaluation System）

手書き文字の評価・管理を行う統合システムです。機械学習データ収集と高精度な文字評価を組み合わせて、日本語手書き文字の美しさを定量的に評価します。

## 🚀 モノリポジトリ構成

このプロジェクトはモノリポジトリとして管理されており、フロントエンドとバックエンドを統合的に開発できます。

### 🛠️ 開発環境セットアップ

```bash
# プロジェクト全体のセットアップ
npm install
npm run setup

# 開発環境の起動（API + Flutter同時起動）
npm run dev

# 個別起動
npm run dev:api      # Python API のみ（Docker）
npm run dev:flutter  # Flutter メインアプリ（WSL→Android）
npm run dev:flutter:basic  # Flutter 基本アプリ（WSL→Android）
npm run dev:flutter:web  # Flutter メインアプリ（Web版）
npm run dev:flutter:basic-web  # Flutter 基本アプリ（Web版）

# Docker管理
npm run docker:up    # APIコンテナ起動
npm run docker:down  # APIコンテナ停止
npm run docker:logs  # APIログ表示
npm run docker:rebuild  # APIコンテナ再ビルド

# テスト実行
npm run test:flutter

# ビルド
npm run build:flutter

# クリーンアップ
npm run clean:flutter
```

**注意**: Python APIはDev Container環境で動作するため、Docker ComposeまたはVSCodeのDev Container機能を使用してください。

## 🎯 プロジェクト概要

このプロジェクトは、手書き文字の評価を効率化し、機械学習に活用可能なデータを収集・管理するためのシステムです。

- **データ管理アプリ** (`moji_manage_app`): Flutter製のモバイル・タブレット対応データ収集アプリ
- **評価API** (`handwriting-eval-api`): Python製の高精度文字評価エンジン

## 📱 システム構成

### 1. 美文字データ管理アプリ (`moji_manage_app`)

Flutter製のクロスプラットフォームアプリケーション

#### 主要機能
- **📸 画像キャプチャ**: スキャナ・カメラでの手書き文字読み込み
- **🔧 自動補正**: OpenCVによるトンボ検出・歪み補正
- **✂️ 文字切り出し**: 自動マス目検出・個別文字保存
- **📊 データ管理**: 切り出し画像のグリッド表示・管理
- **✏️ 評価入力**: 4軸評価（形・黒・白・場）のスコア入力

#### 技術スタック
- **UI**: Flutter (Material Design)
- **カメラ**: `camera` package
- **画像処理**: `image_picker` package
- **ストレージ**: ローカルファイルシステム

### 2. 手書き文字評価API (`handwriting-eval-api`)

Python製の高精度文字評価エンジン

#### 主要機能
- **🔍 4軸評価システム**: 形・黒・白・場の定量的評価
- **🚀 FastAPI**: RESTful API による評価機能提供
- **📈 精密化分析**: 局所ムラ・筆圧・線幅の詳細解析
- **📊 Swagger UI**: 自動生成APIドキュメント
- **🌐 CORS対応**: フロントエンドからの直接アクセス

#### 技術スタック
- **Backend**: FastAPI, uvicorn
- **画像処理**: OpenCV, NumPy, Pillow
- **機械学習**: scikit-image, scipy
- **API**: RESTful, Base64/ファイルアップロード対応

## 🔧 評価システム詳細

### 4軸評価基準

#### 1. 形（Shape Score）- 重み30%
- **スケール対応ハイブリッド形状評価**
- マルチスケール位置補正IoU (70%) + 改良Huモーメント (30%)
- 位置・サイズ不変な形状評価（相似形を適切に評価）

#### 2. 黒（Black Score）- 重み20%
- **線質の包括的評価**
- 線幅安定性 (60%) + 濃淡均一性 (40%)
- 筆圧・線の強さ・安定性の多面的分析

#### 3. 白（White Score）- 重み30%
- **空白バランス評価**
- 黒画素密度の類似度による余白の美しさ評価

#### 4. 場（Center Score）- 重み20%
- **配置位置評価**
- 文字の重心位置による配置バランス評価

### 精密化機能（Phase 1.5）
- **局所ムラ検出**: 15x15スライディングウィンドウ解析
- **エッジ強度評価**: Sobelフィルタによる境界明瞭度
- **筆圧推定**: 複数閾値・ヒストグラム分析
- **方向性考慮**: 縦画・横画・斜線の方向別解析

## 🚀 セットアップ・使用方法

### 必要環境
- Flutter SDK (データ管理アプリ)
- Python 3.8+ (評価API)
- Docker (Dev Container対応)

### 評価API の起動

```bash
# リポジトリのクローン
git clone <repository-url>
cd bimoji-workspace/handwriting-eval-api

# 依存関係のインストール
pip install -r requirements.txt

# APIサーバーの起動
uvicorn api_server:app --reload --host 0.0.0.0 --port 8001

# ブラウザでSwagger UIにアクセス
# http://localhost:8001/docs
```

### Flutter アプリの起動

```bash
cd bimoji-workspace/moji_manage_app

# 依存関係のインストール
flutter pub get

# アプリの実行
flutter run
```

### CLI評価ツール

```bash
# 基本評価
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg

# 精密化評価
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg --enhanced

# JSON出力
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg --enhanced --json
```

## 📊 API使用例

### Base64画像評価
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
    "enhanced_analysis": True
}

response = requests.post("http://localhost:8001/evaluate", json=data)
result = response.json()
print(f"総合スコア: {result['scores']['total']}%")
```

### ファイルアップロード評価
```python
import requests

files = {
    "reference_image": open("reference.jpg", "rb"),
    "target_image": open("user.jpg", "rb")
}
data = {"enhanced_analysis": True}

response = requests.post("http://localhost:8001/evaluate/upload", 
                        files=files, data=data)
result = response.json()
```

## 📈 システム性能

### 評価精度
- **位置ロバスト性**: 同一形状で位置ずれがある場合、100%の精度維持
- **スケールロバスト性**: サイズ違いの相似形評価で89%+の高精度
- **形状識別性**: 異なる形状の適切な区別（円vs正方形: 86.7%）
- **線質評価**: 多面的な線質分析による詳細な問題点検出

### API性能
- **リアルタイム評価**: 高速な評価処理
- **マルチ入力対応**: Base64とファイルアップロード
- **詳細診断**: 改善提案の自動生成

## 🔄 データフロー

1. **データ収集**: Flutterアプリで手書き文字を撮影・管理
2. **前処理**: トンボ検出・歪み補正・文字切り出し
3. **評価**: Python APIで4軸評価実行
4. **結果保存**: JSON形式での評価データ保存
5. **機械学習**: 収集データの学習用データセット化

## 🛠️ 開発環境

### Dev Container対応
VSCodeのDev Container環境で開発可能

```bash
# VSCodeでリポジトリを開く
# F1 > Remote-Containers: Reopen in Container
```

### テスト・検証

```bash
# 形状評価テスト
python validation/shape_evaluation_comparison.py

# 位置ロバスト性テスト
python validation/test_position_robustness.py

# 精密化機能テスト
python validation/test_enhanced_analysis.py
```

## 📋 今後の展開

### Phase 2: 高度な線質解析（計画中）
- リアルタイム筆圧解析
- 筆順推定機能
- 書体スタイル解析

### Phase 3: インテリジェント評価（計画中）
- 機械学習による高度評価
- 3D筆跡解析
- 文脈的評価（複数文字の調和性）

### 基本機能拡張
- 複数文字対応
- 色付き文字対応
- 国際化対応（ひらがな、カタカナ、アルファベット）
- Web版・クラウド同期

## 📁 ディレクトリ構造

```
bimoji-workspace/                 # モノリポジトリルート
├── package.json                 # ワークスペース設定・統合スクリプト
├── README.md                    # このファイル
├── CLAUDE.md                    # 開発ガイド（統合）
├── moji_manage_app/             # Flutter データ管理アプリ
│   ├── lib/                     # Flutter ソースコード
│   ├── docs/                    # アプリ固有ドキュメント
│   └── CLAUDE.md                # 開発ガイド
├── bimoji_app/                  # Flutter 基本アプリ
│   └── lib/                     # Flutter ソースコード
└── handwriting-eval-api/        # Python 評価API
    ├── src/                     # 評価エンジンソース
    ├── tests/                   # テストコード
    ├── validation/              # 検証プログラム
    ├── docs/                    # API仕様書
    ├── data/samples/            # サンプル画像
    └── api_server.py            # FastAPI サーバー
```

## 🤝 貢献・ライセンス

このプロジェクトは、手書き文字評価の研究・教育目的で開発されています。
機械学習・画像処理技術の向上と、日本語手書き文字教育の効率化を目指しています。

---

**最終更新**: 2025-06-15  
**バージョン**: moji_manage_app v1.0.0, handwriting-eval-api v0.2.4