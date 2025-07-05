# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリでコードを扱う際のガイダンスを提供します。

## 最新アップデート (v0.4.0 - 2025-07-05)

### 🚀 Gemini API統合による革新的文字認識システム

**重要な変更**: 座標決め打ちシステムを完全廃止し、AI + 動的検出による次世代システムに進化

#### 主要コマンド
```bash
# Gemini統合版OCRプロセッサ（推奨）
docker exec bimoji-workspace-handwriting-eval-api-1 python src/core/improved_ocr_processor.py

# page_split.py の高精度検出（参考実装）
docker exec bimoji-workspace-handwriting-eval-api-1 python page_split.py docs/記入sample.JPG --out_dir out --dbg

# デバッグ画像確認
docker exec bimoji-workspace-handwriting-eval-api-1 ls debug/
```

#### 環境設定
```bash
# 1. .env にGemini APIキーを設定
cp .env.example .env
# GEMINI_API_KEY=your_actual_api_key_here

# 2. 依存関係インストール
docker exec bimoji-workspace-handwriting-eval-api-1 pip install -r requirements.txt
```

## 開発コマンド（従来）

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

## 🤖 Gemini API統合システム (v0.4.0)

### 概要
手書き文字認識にGoogle Gemini APIを統合し、**99%信頼度**での文字認識を実現。従来の座標決め打ちシステムを廃止し、page_split.pyの高精度動的検出アルゴリズムを統合。

### 主要機能

#### 1. Gemini文字認識クライアント (`src/core/gemini_client.py`)
- **画像変換**: OpenCV → PIL → Base64エンコード
- **プロンプト最適化**: 日本語手書き文字認識に特化
- **構造化出力**: JSON形式での詳細結果
- **エラー処理**: フォールバック機能付き

#### 2. 統合OCRプロセッサ (`src/core/improved_ocr_processor.py`)
- **動的検出**: page_split.pyアルゴリズム統合
- **Gemini統合**: 自動APIクライアント初期化
- **デバッグ管理**: 統一デバッグディレクトリ
- **環境分離**: プロダクション・開発環境対応

### 性能実績（記入sample.JPG）

#### 📝 文字認識（Gemini API）: **100%成功**
- **清**: 99%信頼度 - さんずい + 青の構造認識
- **炎**: 98%信頼度 - 火×2の縦構造認識  
- **葉**: 98%信頼度 - くさかんむり + 世 + 木の詳細分析

#### 🔢 数字認識（Tesseract + 前処理強化）: **83%成功**
- 記入者番号: "7" ✅
- 評価数字: 10/12個読み取り成功（83%）

#### 🎯 動的検出: **100%成功**
- 文字セル検出: 3/3 ✅
- 点数枠検出: 12/12 ✅ 
- コメント枠検出: 12/12 ✅

### 技術仕様

#### Gemini API設定
```python
# 環境変数
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TIMEOUT=30
```

#### プロンプト最適化
```
この手書き文字の画像を解析して、書かれている日本語文字（ひらがな、カタカナ、漢字）を認識してください。

以下の形式でJSONレスポンスを返してください：
{
  "character": "認識した文字",
  "confidence": 0.95,
  "alternatives": ["代替候補1", "代替候補2"],
  "reasoning": "認識の根拠や特徴"
}
```

#### 動的検出アルゴリズム
```python
# 文字セル検出
1. 適応的二値化 → 膨張処理
2. 輪郭抽出 → 四角形フィルタ
3. K-meansクラスタリング → 右列選択
4. Y座標ソート → 3セル抽出

# 点数・コメント枠検出  
1. 右40%領域抽出
2. アスペクト比判定（正方形 vs 横長）
3. 右端列選択 → Y座標ソート
4. 12個ずつ抽出
```

## 従来システム（v0.3.0まで）

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

### 🆕 Gemini統合アーキテクチャ (v0.4.0)

```
Flutter App → FastAPI Server → Gemini API + 動的検出 → データベース・ファイル管理
                ↓
        improved_ocr_processor.py
                ↓
    [動的検出] + [Gemini文字認識] + [Tesseract数字認識]
                ↓
        debug/ ディレクトリ出力（Git除外）
```

#### 核心技術スタック
- **Gemini API**: Google Generative AI (gemini-1.5-flash)
- **動的検出**: page_split.py アルゴリズム統合
- **画像処理**: OpenCV + PIL + Base64変換
- **環境管理**: python-dotenv + .env設定

### 従来アーキテクチャ (v0.3.0まで)

これは**二重構成の手書き文字評価システム**で、異なりながらも補完的な役割を持っています：

#### 1. Flutter データ収集アプリ (`moji_manage_app`)
**アーキテクチャパターン**: ローカルファイルストレージを使用したサービス指向MVC

- **目的**: 機械学習データセット作成のための手書きサンプルの取得、管理、整理
- **コアデータモデル**: `CaptureData` - 記入者ID、タイムスタンプ、文字ラベル、処理ステータスを含む取得サンプルを表現
- **キーサービス**: `CameraService` - カメラ操作、画像取得、ギャラリー選択を処理
- **ストレージ**: `/captures/` 配下のローカルアプリドキュメントディレクトリ、命名規則 `記入者番号_文字_日付.jpg`

**データフロー**:
1. `ImageCaptureScreen` → Camera/Gallery → `CameraService`
2. 画像処理（Gemini統合版：動的検出 + AI認識）
3. 自動文字切り出しと保存
4. `HomeScreen` データ管理ダッシュボード

#### 2. Python 評価エンジン (`handwriting-eval-api`)
**アーキテクチャパターン**: モジュール式スコアリングを使用したパイプライン型評価

##### コアパイプライン (`src/eval/pipeline.py`)
```python
# メイン評価フロー
preprocess_image() → evaluate_4_axes() → weighted_scoring() → detailed_diagnostics()
```

##### 4軸評価システム
- **形（Shape）**: マルチスケール位置補正IoU（70%）+ 改良Huモーメント（30%）
- **黒（Black）**: 線幅安定性（60%）+ 濃淡均一性（40%）
- **白（White）**: ガウス評価による黒画素密度類似度
- **場（Center）**: 重心距離に基づく文字配置

##### 高度機能（拡張モード）
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

### 🆕 Gemini統合処理パイプライン (v0.4.0)

#### 1. 環境初期化・設定
```python
from dotenv import load_dotenv
from gemini_client import GeminiCharacterRecognizer

load_dotenv()
processor = ImprovedOCRProcessor(use_gemini=True, debug_dir="debug")
```

#### 2. 動的検出・透視変換
```python
# page_split.py アルゴリズム統合
corners = find_page_corners(gray)  # 自動ページ検出
warped = perspective_correct_advanced(image, corners)  # 透視変換

# 動的文字セル検出
cells = detect_char_cells(warped_gray)  # 輪郭+K-means
score_boxes, cmt_boxes = detect_score_and_comment_boxes(warped_gray)  # アスペクト比判定
```

#### 3. Gemini文字認識
```python
def recognize_japanese_character(image, context):
    prompt = build_character_recognition_prompt(context)
    response = model.generate_content([prompt, {"mime_type": "image/png", "data": image_base64}])
    return parse_character_response(response.text)

# レスポンス例
{
  "character": "清",
  "confidence": 0.99,
  "alternatives": [],
  "reasoning": "左側の「さんずい（氵）」と右側の「青（せい）」が明確に..."
}
```

#### 4. 統合結果出力
```python
results = {
    "correction_applied": True,
    "character_recognition": {  # Gemini認識結果含む
        "char_1": {
            "image": region_image,
            "bbox": (x1, y1, x2, y2),
            "gemini_recognition": {"character": "清", "confidence": 0.99, ...}
        }
    },
    "writer_number": {...},
    "evaluations": {...},
    "comments": {...},
    "gemini_enabled": True
}
```

### 従来画像処理パイプライン (v0.3.0まで)
1. **前処理**: グレースケール変換 → 台形補正（Hough変換）→ Otsu二値化
2. **形状評価**: 倍率 [0.5, 0.7, 0.8, 1.0, 1.2, 1.4, 2.0] でのマルチスケールテンプレートマッチング
3. **線質評価**: 幅推定のための距離変換 + 濃淡均一性のグレースケール解析
4. **拡張解析**: 選択的統合（基本85% + 拡張濃淡15% + 改良幅10%）

### 性能特性
- **Gemini文字認識**: 日本語漢字で98-99%の高信頼度達成
- **動的検出**: 画像サイズ・品質非依存の安定した領域抽出
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

# Gemini API設定
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_TIMEOUT = 30
```

## 開発ワークフロー

### 🆕 Gemini統合開発フロー (v0.4.0)

#### 1. 新機能開発
```bash
# 1. 環境設定
cp .env.example .env
# GEMINI_API_KEY設定

# 2. 開発
# src/core/gemini_client.py - Gemini API機能
# src/core/improved_ocr_processor.py - 統合処理

# 3. テスト
docker exec bimoji-workspace-handwriting-eval-api-1 python src/core/improved_ocr_processor.py

# 4. デバッグ確認
docker exec bimoji-workspace-handwriting-eval-api-1 ls debug/
```

#### 2. デバッグファイル管理
```bash
# デバッグ画像は debug/ に自動出力
debug/
├── improved_char_char_1.jpg     # 文字画像
├── improved_score_1.jpg         # 点数画像  
├── improved_comment_1.jpg       # コメント画像
├── dbg_cells_contour.jpg        # 検出デバッグ
└── improved_corrected.jpg       # 補正画像

# .gitignore で自動除外
debug/
*.jpg
*.png
```

### 従来開発フロー

#### 新しい評価機能の追加
1. `src/eval/metrics.py` でコアアルゴリズムを実装
2. `tests/test_metrics.py` に単体テストを追加
3. `validation/` ディレクトリに検証スクリプトを作成
4. `src/eval/pipeline.py` のパイプラインを更新
5. `src/eval/cli.py` にCLIオプションを追加
6. `api_server.py` のAPIエンドポイントを更新

#### テスト戦略
- **単体テスト**: pytestを使用したモジュールレベルの機能テスト
- **統合テスト**: エンドツーエンドのパイプライン検証
- **性能検証**: 既知の画像ペアでのロバスト性テスト
- **APIテスト**: サンプルデータを使用したHTTPエンドポイント検証

### Flutter アプリ開発
- **カメラ統合**: 画面間で一貫した画像取得のために `CameraService` を使用
- **データ管理**: すべての手書きサンプルで `CaptureData` モデルパターンに従う
- **ローカルストレージ**: 一貫した命名規則でアプリドキュメントに画像を保存
- **API統合**: Gemini統合版Python評価バックエンドへのHTTP呼び出し

## 重要な開発注意事項

### 🆕 Gemini API統合注意事項 (v0.4.0)

#### API制限・エラー処理
- **レート制限**: デフォルト60req/min（環境変数で調整可能）
- **タイムアウト**: 30秒（調整可能）
- **フォールバック**: API無効時の安全な動作継続
- **エラーログ**: 詳細なエラー情報記録

#### セキュリティ
- **APIキー**: .envファイルで管理、Gitコミット除外
- **画像データ**: Base64エンコードで安全な送信
- **ログ機密性**: APIキーの誤ログ出力防止

#### デバッグファイル管理
- **自動出力**: debug/ディレクトリに統一
- **Git除外**: .gitignore で版管理クリーンアップ
- **環境分離**: プロダクション・開発環境対応

### 従来注意事項

#### 画像フォーマット処理
- OpenCVはBGRを使用、Web・モバイルは通常RGBを使用 - APIで自動変換実装済み
- すべての評価は8ビットグレースケールまたはBGRカラー画像を期待
- バイナリマスクは0（背景）と255（前景）を使用

#### 日本語UI コンテキスト
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

## 🆕 次世代機能ロードマップ (v0.5.0+)

### Phase 1: Gemini機能拡張
- [ ] コメント欄OCR（日本語テキスト認識）
- [ ] 一括文字認識API（複数画像同時処理）
- [ ] 認識精度向上（プロンプトエンジニアリング）
- [ ] 代替モデル対応（Claude、GPT-4V等）

### Phase 2: Flutter統合
- [ ] Gemini認識結果確認・訂正UI
- [ ] リアルタイム文字認識プレビュー  
- [ ] オフライン対応（モデル軽量化）
- [ ] 使用量・コスト管理ダッシュボード

### Phase 3: MLOps・プロダクション
- [ ] 認識精度モニタリング
- [ ] A/Bテスト（複数モデル比較）
- [ ] 自動再学習パイプライン
- [ ] エッジデバイス最適化

## 従来ロードマップ（OCRフォーム処理システム）

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
# 最新版実行（Gemini統合）
docker exec bimoji-workspace-handwriting-eval-api-1 python src/core/improved_ocr_processor.py

# 改良点:
# - 複数前処理手法の並列適用
# - 文字と数字の特化型OCR
# - デバッグ画像自動生成
# - Gemini API統合による高精度文字認識
```

### 処理フロー詳細

#### Phase 1: 動的検出（Gemini統合版）
```python
# 自動ページ検出
corners = find_page_corners(gray)

# 文字セル検出（輪郭+K-means）
cells = detect_char_cells(warped_gray)

# 点数・コメント枠検出（アスペクト比判定）
score_boxes, cmt_boxes = detect_score_and_comment_boxes(warped_gray)
```

#### Phase 2: 透視変換
```python
# アスペクト比計算
wA = np.linalg.norm(pts[2] - pts[3])
wB = np.linalg.norm(pts[1] - pts[0])
W = int(max(wA, wB))

# 変換行列生成
transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
corrected = cv2.warpPerspective(image, transform_matrix, (W, H))
```

#### Phase 3: 領域抽出・認識

**文字領域（Gemini認識）**:
```python
character_results = {}
for region_image in character_regions:
    gemini_result = gemini_client.recognize_japanese_character(region_image)
    # character, confidence, alternatives, reasoning
```

**数字領域（Tesseract + 前処理強化）**:
```python
# 複数前処理手法
binary_methods = ["otsu", "adaptive_mean", "adaptive_gaussian", "manual_light", "manual_dark"]

# 複数OCR設定
ocr_configs = [
    '--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789',
    '--oem 3 --psm 6',  # ホワイトリストなし
]
```

### 性能指標 (記入sample.JPG - Gemini統合版)

#### 文字画像切り出し: ✅ 100% (3/3)
- char_1: 高品質画像保存 → Gemini認識: "清" (99%)
- char_2: 高品質画像保存 → Gemini認識: "炎" (98%)
- char_3: 高品質画像保存 → Gemini認識: "葉" (98%)

#### 数字OCR: ✅ 約83% (10/12)
- 記入者番号: 成功 ("7" 読み取り)
- 評価数字: 10個読み取り可能

### デバッグ・トラブルシューティング

#### 生成される画像ファイル
```bash
# 文字画像 (高精度)
debug/improved_char_char_1.jpg
debug/improved_char_char_2.jpg  
debug/improved_char_char_3.jpg

# 数字画像 (OCR用)
debug/improved_score_1.jpg
debug/improved_score_2.jpg
# ... など

# 前処理デバッグ画像
debug/improved_debug_記入者番号_otsu.jpg
debug/improved_debug_白評価1_manual_light.jpg
# ... など

# 検出デバッグ画像
debug/dbg_cells_contour.jpg     # 文字セル検出
debug/dbg_score_comment_boxes.jpg  # 点数・コメント枠検出
```

#### よくある問題と対処法

1. **Gemini API認識失敗**
   - 原因: APIキー未設定、ネットワーク問題
   - 対処: .env設定確認、フォールバック動作確認

2. **動的検出失敗**
   - 原因: 画像品質、フォーム形式の違い
   - 対処: debug/画像でプロセス確認、閾値調整

3. **OCR精度低下**
   - 原因: 前処理不適切、フォント認識
   - 対処: 複数前処理手法の並列実行確認

### 次回開発予定

#### 優先度: 高
1. **API統合**: FastAPIエンドポイント作成
2. **Flutter連携**: Gemini認識結果確認UI実装
3. **精度改善**: コメント欄日本語OCR追加

#### 優先度: 中
1. **バッチ処理**: 複数記入用紙の一括処理
2. **学習データ**: 認識訂正結果の蓄積システム
3. **性能最適化**: 処理時間短縮・コスト最適化

### 開発環境セットアップ

```bash
# Docker環境での開発
npm run docker:up

# Gemini統合版テスト実行
docker exec bimoji-workspace-handwriting-eval-api-1 python src/core/improved_ocr_processor.py

# デバッグ画像ファイル確認
docker exec bimoji-workspace-handwriting-eval-api-1 ls debug/

# デバッグ画像の取得
docker cp bimoji-workspace-handwriting-eval-api-1:/app/debug/improved_char_char_1.jpg /tmp/
```

## 最終更新情報

**最終更新**: 2025-07-05  
**更新者**: Claude Code  
**主要変更**: Gemini API統合、動的検出システム、デバッグファイル管理統一

**現在の推奨実行環境**:
- Docker: ✅ 対応
- Gemini API: ✅ 統合済み  
- デバッグ機能: ✅ debug/ディレクトリ統一
- Git管理: ✅ クリーンアップ完了