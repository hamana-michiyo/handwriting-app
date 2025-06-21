# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリでコードを扱う際のガイダンスを提供します。

## 開発コマンド

### モノリポジトリ全体の起動（推奨）

#### プロジェクト初期設定
```bash
# 依存関係のインストール
npm install
npm run setup

# 開発環境の起動（API + Flutter同時起動）
npm run dev

# 個別起動コマンド
npm run dev:api            # Python API のみ（Docker）
npm run dev:flutter        # Flutter メインアプリ（WSL→Android）
npm run dev:flutter:basic  # Flutter 基本アプリ（WSL→Android）
npm run dev:flutter:web    # Flutter メインアプリ（Web版）
npm run dev:flutter:basic-web  # Flutter 基本アプリ（Web版）
```

#### Docker管理
```bash
npm run docker:up          # APIコンテナ起動
npm run docker:down        # APIコンテナ停止
npm run docker:logs        # APIログ表示
npm run docker:rebuild     # APIコンテナ再ビルド
```

#### テスト・ビルド・クリーンアップ
```bash
npm run test:flutter       # Flutterテスト実行
npm run build:flutter      # Flutterビルド
npm run clean:flutter      # Flutterクリーンアップ
npm run lint:flutter       # Flutter静的解析
```

### Python API Server (handwriting-eval-api)

#### サーバー起動
```bash
cd handwriting-eval-api
# 開発サーバー（自動リロード）
uvicorn api_server:app --reload --host 0.0.0.0 --port 8001

# Swagger UI ドキュメントへのアクセス
# http://localhost:8001/docs
```

#### CLI評価ツール
```bash
cd handwriting-eval-api
# 基本評価
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg

# 精密化評価（Phase 1.5機能）
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg --enhanced

# JSON出力形式
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg --enhanced --json

# デバッグモード（中間画像出力）
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg --dbg
```

#### テストコマンド
```bash
cd handwriting-eval-api
# 単体テスト
pytest
pytest tests/test_metrics.py -v

# 性能検証スクリプト
python validation/shape_evaluation_comparison.py
python validation/test_position_robustness.py
python validation/test_scale_robustness.py
python validation/test_enhanced_analysis.py
```

### Flutter アプリ

#### メインデータ管理アプリ (moji_manage_app)
```bash
cd moji_manage_app
flutter pub get           # 依存関係のインストール
flutter run              # 接続デバイス・エミュレータで実行
flutter test             # テスト実行
flutter analyze          # 静的解析
flutter clean            # ビルドキャッシュクリア

# プラットフォーム別ビルド
flutter build apk        # Android APK
flutter build ios        # iOS（macOS必須）
flutter build web        # Web版
```

#### 基本アプリ (bimoji_app)
```bash
cd bimoji_app
flutter pub get && flutter run
```

## プロジェクトアーキテクチャ

これは**二重構成の手書き文字評価システム**で、異なりながらも補完的な役割を持っています：

### 1. Flutter データ収集アプリ (`moji_manage_app`)
**アーキテクチャパターン**: ローカルファイルストレージを使用したサービス指向MVC

- **目的**: 機械学習データセット作成のための手書きサンプルの取得、管理、整理
- **コアデータモデル**: `CaptureData` - 記入者ID、タイムスタンプ、文字ラベル、処理ステータスを含む取得サンプルを表現
- **キーサービス**: `CameraService` - カメラ操作、画像取得、ギャラリー選択を処理
- **ストレージ**: `/captures/` 配下のローカルアプリドキュメントディレクトリ、命名規則 `記入者番号_文字_日付.jpg`

**データフロー**:
1. `ImageCaptureScreen` → Camera/Gallery → `CameraService`
2. 画像処理（計画中：Python バックエンド経由の OpenCV 歪み補正）
3. 自動文字切り出しと保存
4. `HomeScreen` データ管理ダッシュボード

### 2. Python 評価エンジン (`handwriting-eval-api`)
**アーキテクチャパターン**: モジュール式スコアリングを使用したパイプライン型評価

#### コアパイプライン (`src/eval/pipeline.py`)
```python
# メイン評価フロー
preprocess_image() → evaluate_4_axes() → weighted_scoring() → detailed_diagnostics()
```

#### 4軸評価システム
- **形（Shape）**: マルチスケール位置補正IoU（70%）+ 改良Huモーメント（30%）
- **黒（Black）**: 線幅安定性（60%）+ 濃淡均一性（40%）
- **白（White）**: ガウス評価による黒画素密度類似度
- **場（Center）**: 重心距離に基づく文字配置

#### 高度機能（拡張モード）
- **局所テクスチャ解析**: 局所密度変動のための15x15スライディングウィンドウ
- **エッジ強度評価**: 境界明瞭度評価のためのSobelフィルタ
- **多閾値筆圧推定**: 筆圧解析のためのヒストグラム解析
- **方向性ストローク解析**: 縦画・横画・斜線の方向別評価

### 3. FastAPI 統合レイヤー
フロントエンド統合のための CORS 対応 **RESTful API**:
- `POST /evaluate` - Base64画像評価
- `POST /evaluate/upload` - ファイルアップロード評価
- `GET /docs` - Swagger UIドキュメント
- OpenCV互換性のための自動BGR↔RGB変換

## 技術実装詳細

### 画像処理パイプライン
1. **前処理**: グレースケール変換 → 台形補正（Hough変換）→ Otsu二値化
2. **形状評価**: 倍率 [0.5, 0.7, 0.8, 1.0, 1.2, 1.4, 2.0] でのマルチスケールテンプレートマッチング
3. **線質評価**: 幅推定のための距離変換 + 濃淡均一性のグレースケール解析
4. **拡張解析**: 選択的統合（基本85% + 拡張濃淡15% + 改良幅10%）

### 性能特性
- **位置ロバスト性**: 位置オフセットのある同一形状で100%スコア維持
- **スケールロバスト性**: 異なるサイズの類似形状で89%+の精度
- **形状識別性**: 異なる形状の適切な区別（円vs正方形: 86.7%）

### 主要設定パラメータ
```python
# スコアリング重み
SHAPE_W = 0.30, BLACK_W = 0.20, WHITE_W = 0.30, CENTER_W = 0.20

# 形状評価
SHAPE_IOU_WEIGHT = 0.7, SHAPE_HU_WEIGHT = 0.3
SCALE_FACTORS = [0.5, 0.7, 0.8, 1.0, 1.2, 1.4, 2.0]

# 線質評価
BLACK_WIDTH_WEIGHT = 0.6, BLACK_INTENSITY_WEIGHT = 0.4
```

## 開発ワークフロー

### 新しい評価機能の追加
1. `src/eval/metrics.py` でコアアルゴリズムを実装
2. `tests/test_metrics.py` に単体テストを追加
3. `validation/` ディレクトリに検証スクリプトを作成
4. `src/eval/pipeline.py` のパイプラインを更新
5. `src/eval/cli.py` にCLIオプションを追加
6. `api_server.py` のAPIエンドポイントを更新

### テスト戦略
- **単体テスト**: pytestを使用したモジュールレベルの機能テスト
- **統合テスト**: エンドツーエンドのパイプライン検証
- **性能検証**: 既知の画像ペアでのロバスト性テスト
- **APIテスト**: サンプルデータを使用したHTTPエンドポイント検証

### Flutter アプリ開発
- **カメラ統合**: 画面間で一貫した画像取得のために `CameraService` を使用
- **データ管理**: すべての手書きサンプルで `CaptureData` モデルパターンに従う
- **ローカルストレージ**: 一貫した命名規則でアプリドキュメントに画像を保存
- **将来のAPI統合**: Python評価バックエンドへのHTTP呼び出し設計

## 重要な開発注意事項

### 画像フォーマット処理
- OpenCVはBGRを使用、Web・モバイルは通常RGBを使用 - APIで自動変換実装済み
- すべての評価は8ビットグレースケールまたはBGRカラー画像を期待
- バイナリマスクは0（背景）と255（前景）を使用

### 日本語UI コンテキスト
Flutterアプリは日本語手書き文字評価を対象としているため全体で日本語を使用：
- 記入者番号 = Writer ID/Number
- 美文字 = Beautiful handwriting  
- 評価 = Evaluation
- 形・黒・白・場 = Shape, Black (ink), White (spacing), Center (positioning)

### Dev Container サポート
両コンポーネントはVSCode Dev Container開発をサポート：
- すべての依存関係がプリインストールされたPython環境
- Flutter SDKとプラットフォームツールが設定済み
- 実験的開発のためのJupyter notebookサポート