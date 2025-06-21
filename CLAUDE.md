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

## OCRフォーム処理システム (2025-06-21 実装)

### 概要
記入用紙からの手書き文字・数字自動抽出システム。トンボ（位置合わせマーク）検出による高精度補正と、機械学習データセット用文字画像切り出しを実現。

### 主要コンポーネント

#### 1. トンボ検出・歪み補正 (`src/core/fixed_form_processor.py`)
```bash
# 基本実行
docker exec bimoji-workspace-handwriting-eval-api-1 python src/core/fixed_form_processor.py

# 機能:
# - 4点トンボ自動検出 (座標範囲指定)
# - 透視変換による歪み補正
# - 正確なアスペクト比計算
```

#### 2. ハイブリッド処理 (`src/core/hybrid_processor.py`)
```bash
# 実行方法
docker exec bimoji-workspace-handwriting-eval-api-1 python src/core/hybrid_processor.py

# 特徴:
# - 文字: トンボ補正後の高精度切り出し
# - 数字: 元画像からの直接OCR
# - 用途別最適化処理
```

#### 3. 改良版OCR処理 (`src/core/improved_ocr_processor.py`) - **推奨**
```bash
# 最新版実行
docker exec bimoji-workspace-handwriting-eval-api-1 python src/core/improved_ocr_processor.py

# 改良点:
# - 複数前処理手法の並列適用
# - 文字と数字の特化型OCR
# - デバッグ画像自動生成
```

### 処理フロー詳細

#### Phase 1: トンボ検出
```python
# 検出対象座標 (元画像ベース)
tombo_regions = [
    (540, 190, 550, 200),   # 左上
    (890, 190, 900, 200),   # 右上
    (540, 1400, 550, 1410), # 左下
    (890, 1400, 900, 1410)  # 右下
]

# 検出条件
- サイズ範囲: 8-30ピクセル
- 面積範囲: 50-900平方ピクセル
- 形状スコア: 円形度 × アスペクト比
```

#### Phase 2: 透視変換
```python
# アスペクト比計算
avg_width = (width_top + width_bottom) // 2
avg_height = (height_left + height_right) // 2
target_width = int(1200 * avg_width / avg_height)

# 変換行列生成
transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
corrected = cv2.warpPerspective(image, transform_matrix, (target_width, target_height))
```

#### Phase 3: 領域抽出

**文字領域 (補正画像から相対座標)**:
```python
character_coords = [
    ("清", 600, 810, 280, 470),   # x1, x2, y1, y2
    ("炎", 600, 810, 700, 900), 
    ("葉", 600, 810, 1110, 1310)
]

# トンボ領域内相対座標変換
tombo_area = {545, 195, 895, 1405}  # x1, y1, x2, y2
x1_rel = (x1 - 545) / (895 - 545)
```

**数字領域 (元画像から絶対座標)**:
```python
number_regions = [
    ("記入者番号", 1800, 2300, 100, 170),
    ("白評価1", 2220, 2330, 200, 300),
    # ... 12個の評価数字領域
]
```

#### Phase 4: OCR処理

**数字用強化前処理**:
```python
preprocessing_methods = [
    "otsu",              # Otsu二値化
    "adaptive_mean",     # 適応的平均二値化  
    "adaptive_gaussian", # 適応的ガウシアン二値化
    "manual_light",      # 手動閾値(明)
    "manual_dark"        # 手動閾値(暗)
]

ocr_configs = [
    '--oem 3 --psm 8',   # 単語レベル
    '--oem 3 --psm 10',  # 単一文字
    '--oem 1 --psm 8'    # LSTM OCR
]
```

### 性能指標 (記入sample.JPG)

#### 文字画像切り出し: ✅ 100% (3/3)
- 清: 高品質画像保存
- 炎: 高品質画像保存  
- 葉: 高品質画像保存

#### 数字OCR: ✅ 約50% (6/12)
- 記入者番号: 部分的成功 ("No. 3" → "7" 誤認識)
- 評価数字: 約半数読み取り可能

### デバッグ・トラブルシューティング

#### 生成される画像ファイル
```bash
# 文字画像 (高精度)
improved_char_清.jpg
improved_char_炎.jpg  
improved_char_葉.jpg

# 数字画像 (OCR用)
improved_num_記入者番号.jpg
improved_num_白評価1.jpg
# ... など

# 前処理デバッグ画像
improved_debug_記入者番号_otsu.jpg
improved_debug_記入者番号_adaptive_mean.jpg
# ... など
```

#### よくある問題と対処法

1. **トンボ検出失敗**
   - 原因: 画像品質、照明条件
   - 対処: 検出範囲拡大、閾値調整

2. **文字切り出し位置ズレ**
   - 原因: 透視変換のアスペクト比誤計算
   - 対処: トンボ間距離の再計算

3. **OCR精度低下**
   - 原因: 前処理不適切、フォント認識
   - 対処: 複数前処理手法の並列実行

### 次回開発予定

#### 優先度: 高
1. **API統合**: FastAPIエンドポイント作成
2. **Flutter連携**: OCR結果確認UI実装
3. **精度改善**: PaddleOCR統合検討

#### 優先度: 中
1. **バッチ処理**: 複数記入用紙の一括処理
2. **学習データ**: OCR訂正結果の蓄積システム
3. **性能最適化**: 処理時間短縮

### 開発環境セットアップ

```bash
# Docker環境での開発
npm run docker:up

# OCRテスト実行
docker exec bimoji-workspace-handwriting-eval-api-1 python src/core/improved_ocr_processor.py

# 画像ファイル確認
docker exec bimoji-workspace-handwriting-eval-api-1 ls improved_*.jpg

# デバッグ画像の取得
docker cp bimoji-workspace-handwriting-eval-api-1:/app/improved_char_清.jpg /tmp/
```