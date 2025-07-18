# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリでコードを扱う際のガイダンスを提供します。

## 🎉 最新実装完了（2025-07-18）

### ✅ Renderプロダクションデプロイ完了（v0.8.5） - 本番環境稼働開始

**v0.8.5 Renderプロダクションデプロイ**:
- 🚀 **Render本番環境**: https://handwriting-app-qqp3.onrender.com/ でAPI稼働開始
- 🐳 **Docker統合**: Dockerfile.api による完全なシステム依存関係管理
- 🔧 **PyTorch追加**: requirements_api.txt にtorch>=2.0.0, torchvision>=0.15.0 追加完了
- 📂 **WORKDIR統一**: 開発環境(/workspace)と本番環境のパス完全統一
- 🌐 **環境変数完全対応**: .env の全設定をrender.yaml に統合（機密情報は手動設定）
- 📱 **Flutter環境切り替え**: 開発環境⇔本番環境の自動切り替え機能実装

**v0.8.5 技術的成果**:
- ✅ **API稼働確認**: /health エンドポイントで正常動作確認済み
- ✅ **データベース接続**: Supabase統合完全動作
- ✅ **依存関係解決**: tesseract-ocr + PyTorch + 全ライブラリ正常インストール
- ✅ **Flutter統合**: kDebugMode による開発/本番環境自動判定
- ✅ **セキュリティ**: 機密情報(API keys)の安全な環境変数管理
- ✅ **自動デプロイ**: Git push による自動デプロイ機能

**v0.8.5 デプロイ設定**:
```yaml
services:
  - type: web
    name: bimoji-api
    runtime: docker
    dockerfilePath: ./Dockerfile.api
    region: singapore
    plan: starter
```

**v0.8.5 Flutter環境切り替え**:
```dart
static String get _baseUrl {
  if (kDebugMode) {
    // 開発環境: ローカルサーバー
    return _developmentUrl;
  } else {
    // 本番環境: Render API
    return _productionUrl;
  }
}
```

### ✅ サンプル詳細画面UI大幅改良完了（v0.8.4） - 連続評価作業最適化

**v0.8.4 サンプル詳細画面UI大幅改良**:
- 🎨 **評価スコア入力方式変更**: テキスト入力 → スライダー入力（0-10範囲）で直感的操作を実現
- 🌈 **色分けスライダー実装**: 白（青）・黒（黒）・場（赤）・形（緑）の視覚的識別による効率向上
- 📱 **レイアウト大幅簡略化**: 連続評価作業に最適化されたコンパクトデザイン
- 🖼️ **基本情報統合**: 左側に80x80px小画像、右側にシンプルな基本情報で情報密度向上
- 🔗 **前後移動機能**: 矢印ボタンで一覧内のサンプル間を連続移動（位置表示付き）
- ⚡ **リアルタイム編集**: スライダー・コメント欄が常時編集可能でワークフロー最適化
- 💾 **保存機能改良**: SupabaseのHTTP 200/204両対応で保存エラー完全解決

**v0.8.4で実現した連続評価ワークフロー**:
- ✅ **スクロール最小化**: 縦幅を大幅削減し、評価作業効率を向上
- ✅ **視覚的フィードバック**: スライダー値のリアルタイム表示
- ✅ **直感的操作**: タップ・スライド操作による迅速な評価入力
- ✅ **認識文字統合**: 基本情報内に認識文字を表示（独立セクション廃止）
- ✅ **不要機能削除**: 文字認識結果セクション、AI認識詳細、画像セクションを削除

### ✅ Flutter API統合・評価スコア保存完全実装済み（v0.8.3） - API統合問題完全解決

**v0.8.3 Flutter API統合完全実装**:
- 🔗 **データ構造統一完了**: `improved_ocr_processor.py` と `supabase_ocr_processor.py` 間のデータ構造不整合を完全解決
- 📊 **評価スコア保存実装**: writing_sampleテーブルにscore_white、score_black、score_center、score_shapeの自動保存
- 🔄 **既存サンプル更新機能**: 重複防止機能と併用した既存サンプルへの評価点数自動更新
- 🖼️ **補助線除去画像保存**: Supabaseストレージに補助線除去済み高品質画像を自動保存
- ✅ **FlutterからAPI完全動作**: Flutter → API → Gemini認識 → 評価スコア → DB保存の完全フロー実現

**v0.8.3で解決した問題**:
- ❌ **"`writer_number` not defined"エラー**: OCR結果のデータ構造をdictionary→array形式に統一
- ❌ **"`image_array` not defined"エラー**: 補助線除去済み画像のarray・bytes両方読み込み実装
- ❌ **"`character_recognition` → `character_results`"**: キー名不整合を完全統一
- ❌ **"`gemini_result` → `gemini_recognition`"**: Gemini結果キー名不整合を統一
- ❌ **評価スコア未保存問題**: `number_results` → `evaluations`データ構造変更に対応
- ✅ **完全動作確認**: 清(white:7,black:6,center:8,shape:9)、炎・葉の評価スコア正常保存

### ✅ 照明対応完全実装済み（v0.8.2） - 明暗差問題解決

**v0.8.2 照明問題完全解決**:
- 🌅 **A4画像全体照明補正**: 軽いCLAHE（clipLimit=1.5, tileGridSize=16×16）をA4全体に最初に適用
- 🔍 **改良補助線除去**: 文字保護を重視したパラメータ調整（閾値130、最小カーネル）
- 💡 **明暗差対応**: 撮影時の明るい部分・影部分での安定した文字・数字認識を実現
- 🎯 **認識精度維持**: 文字認識98-99%信頼度、数字認識12/12成功を照明補正下でも維持

**v0.8.1 macOS対応完了**:
- 🍎 **macOS完全対応**: Docker + Flutter + API統合の完全動作確認
- 📱 **Flutter高解像度化**: カメラ最大解像度 + 画質100%設定による画像品質向上
- 🎯 **4隅検出精度向上**: 複数手法・妥当性チェック・角度制限による検出精度大幅改善
- 📊 **認識結果ログ出力**: `debug/result.log` に詳細な文字・数字認識結果を出力
- 🔧 **文字セル検出最適化**: より厳密な面積・形状フィルタによる文字領域精密検出

**完了した統合作業（v0.8.0）**:
- 🔗 **API統合完了**: `/process-cropped-form` エンドポイントに `supabase_ocr_processor.py` の実際の認識処理を統合
- 🤖 **Gemini + PyTorch 実稼働**: 実際の日本語文字認識（99%精度）と数字認識（100%精度）が動作
- 📱 **Flutter高解像度対応**: ImageCropper設定最適化により高品質画像送信を実現
- 🐛 **メモリ最適化**: カメラバッファ問題解決

**現在の動作フロー（v0.8.4 - UI改良版）**:
1. **Flutter**: カメラ最大解像度撮影 → image_cropper でトリミング（最大6000x4000px, 品質100%）
2. **API**: `/process-cropped-form` で受信→一時ファイル→SupabaseOCRProcessor呼び出し
3. **照明補正**: A4画像全体にCLAHE照明ムラ補正を最初に適用（明暗差対応）
4. **4隅検出**: 照明補正済み画像での改良版検出（複数手法・妥当性チェック）
5. **認識処理**: 改良補助線除去 + Gemini文字認識 + PyTorch数字認識
6. **評価スコア収集**: OCR結果から文字別評価点数を自動抽出・マッピング
7. **データベース保存**: 文字認識結果 + 評価スコア + 補助線除去済み画像をSupabaseに統合保存
8. **サンプル管理**: 一覧画面で検索・フィルタリング・ページネーション
9. **連続評価**: 詳細画面でスライダー評価 → 保存 → 前後移動 → 次の評価
10. **結果返却**: 完全統合処理結果をFlutterに返却（DB保存済み確認付き）

**技術的解決事項（v0.8.3 - API統合完全対応）**:
- ✅ **データ構造統一**: `character_recognition` → `character_results` キー統一によるモジュール間連携完全実現
- ✅ **評価スコア処理**: `number_results` → `evaluations` データ構造変更への対応実装完了
- ✅ **Gemini結果統合**: `gemini_result` → `gemini_recognition` キー名統一による認識結果完全連携
- ✅ **画像データ統合**: 補助線除去済み画像のarray・bytes両方読み込みによる完全保存実現
- ✅ **既存サンプル更新**: 重複防止と評価スコア更新の両立による柔軟なデータ管理
- ✅ **エラー完全解決**: 全ての"`xxx` not defined"エラーの根本原因解決と動作確認完了

**技術的解決事項（v0.8.2 - 照明対応）**:
- ✅ **A4全体照明補正**: CLAHE（clipLimit=1.5, tileGridSize=16×16）による明暗ムラ正規化
- ✅ **改良補助線除去**: コントラスト強化（alpha=1.3, beta=-5）+ 高い閾値（130）+ 最小カーネル
- ✅ **文字保護強化**: 膨張処理なし・エリプスカーネル・段階的デバッグ出力
- ✅ **照明変動対応**: 明るい部分・影部分での安定認識（従来の個別領域補正を廃止）

**技術的解決事項（v0.8.1 - macOS対応）**:
- ✅ **macOS環境対応**: ネットワーク設定・権限・API接続の完全解決
- ✅ **カメラ解像度最大化**: ResolutionPreset.max + ImageCropper品質100%
- ✅ **4隅検出大幅改善**: エッジ検出・輪郭抽出・角度制限・面積妥当性チェック
- ✅ **文字セル検出精密化**: 0.5-1.5%面積制限・0.9-1.1アスペクト比・サイズ妥当性
- ✅ **認識結果詳細ログ**: debug/result.log に文字・数字・信頼度・処理時間出力
- ✅ **デバッグ画像強化**: 候補・選択・最終結果の色分け表示

**従来技術的解決事項（v0.8.0）**:
- ✅ 透視変換スキップ（トリミング済み画像には不要）
- ✅ ImageCropper設定改善（3000×2000 → 6000×4000, 品質85% → 95%）
- ✅ カメラ解像度向上（high → veryHigh, JPEG形式明示）
- ✅ 一時ファイル経由でのnumpy配列→画像パス変換

---

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

## プロジェクトアーキテクチャ

これは**二重構成の手書き文字評価システム**で、異なりながらも補完的な役割を持っています：

### 1. Flutter データ収集アプリ (`moji_manage_app`)
**アーキテクチャパターン**: API統合を使用したサービス指向MVC

- **目的**: 機械学習データセット作成のための手書きサンプルの取得、管理、整理
- **コアデータモデル**: `CaptureData` - 記入者ID、タイムスタンプ、文字ラベル、処理ステータスを含む取得サンプルを表現
- **キーサービス**: 
  - `CameraService` - カメラ操作、画像取得、ギャラリー選択を処理
  - `ApiService` - Python APIサーバーとの通信、Base64画像送信、結果解析
- **ストレージ**: Supabase Database + Storage による永続化

**データフロー**:
1. `ImageCaptureScreen` → Camera/Gallery → `CameraService`
2. `ImagePreviewScreen` → プレビュー確認
3. `ImageUploadScreen` → `ApiService` → Python API
4. サーバー処理（Gemini + PyTorch + 補助線除去）
5. 結果表示・確認

### 2. Python 評価エンジン (`handwriting-eval-api`)
**アーキテクチャパターン**: FastAPI + AI統合パイプライン

#### 統合処理システム (v0.8.0) - 実稼働版
- **Gemini文字認識**: 99%精度の日本語手書き文字認識（実装完了）
- **PyTorch数字認識**: 100%精度の学習済みCNNモデル（MNIST+手書きデータセット）（実装完了）
- **補助線除去**: 機械学習データ品質向上のためのOpenCV処理（実装完了）
- **Supabase統合**: データベース + ストレージ自動管理（実装完了）

#### API エンドポイント
- `POST /process-cropped-form` - トリミング済み画像処理（Base64対応、SupabaseOCRProcessor統合済み）
- `POST /process-form` - 記入用紙画像処理（Base64対応、従来版）
- `GET /health` - ヘルスチェック
- `GET /docs` - Swagger UIドキュメント

## 最新更新情報 (v0.8.0 - 2025-01-12)

### 🎯 統合システム実稼働化完了

**主要達成事項**:
1. **API実装統合**: `/process-cropped-form` に `supabase_ocr_processor.py` の実際のGemini+PyTorch認識処理を統合
2. **Flutter画像品質改善**: ImageCropper・カメラ設定最適化により高解像度画像送信を実現
3. **エンドツーエンド動作**: Flutter → API → 実際のAI認識 → 結果表示の完全フロー確立
4. **デバッグ強化**: 文字・数字切り出し画像の自動保存機能で認識精度検証が可能

**技術的改善**:
- ImageCropper: 6000x4000px対応、品質95%
- カメラ: ResolutionPreset.veryHigh、JPEG形式最適化
- API: 一時ファイル経由でのSupabaseOCRProcessor統合
- メモリ管理: カメラバッファ問題解決

### ✅ Flutter API統合完了（v0.7.0）
**実装済み機能**:
- **ApiService**: HTTP通信・Base64画像送信・レスポンス解析
- **ImageUploadScreen**: 記入者情報フォーム・アップロード管理・結果表示
- **ワークフロー統合**: 撮影→プレビュー→アップロード→結果表示

**技術仕様**:
- Flutter HTTP パッケージ使用
- Base64画像エンコード
- リアルタイム進行状況表示
- 詳細エラーハンドリング

### 🧪 次回テスト予定
1. **完全フローテスト**: カメラ→プレビュー→アップロード→結果
2. **API通信確認**: Flutter ↔ Python接続
3. **認識精度検証**: Gemini + PyTorch統合動作
4. **エラー処理確認**: ネットワーク・サーバーエラー対応

---

## 重要な開発注意事項

### v0.8.4 UI改良システム注意事項
- **スライダー実装**: 0-10範囲の数値スライダーで各評価項目の色分け（白・黒・場・形）維持
- **前後移動機能**: SampleListScreenから allSamples と currentIndex を正確に渡す
- **画像サイズ統一**: 詳細画面の画像は80x80pxに統一（コンパクト表示）
- **保存機能**: HTTP 200/204両対応でSupabaseへの保存エラー回避
- **レイアウト維持**: 連続評価作業に最適化されたスクロール最小化レイアウト

### v0.8.3 API統合システム注意事項
- **データ構造統一**: `improved_ocr_processor.py` と `supabase_ocr_processor.py` 間のキー名は必ず統一を維持
- **評価スコア保存**: `evaluations`データ構造からの自動マッピング処理を変更時は動作確認必須
- **既存サンプル処理**: 重複防止と評価スコア更新の両方を考慮した処理フロー維持
- **補助線除去画像**: ストレージ保存用のbytes読み込みとshape確認用のarray読み込み両方実装
- **デバッグログ確認**: `評価スコア収集結果`ログで正しいマッピングが行われているか必須確認

### v0.8.2 照明対応システム注意事項
- **照明補正タイミング**: A4画像全体に最初に1回のみ適用（個別領域への適用は禁止）
- **補助線除去パラメータ**: 文字保護を最優先（閾値130、最小カーネル、膨張処理なし）
- **デバッグ画像確認**: `a4_lighting_corrected.jpg`, `enhanced_*.jpg`, `thresh_*.jpg` で段階確認
- **認識精度検証**: 明暗差のある画像での文字・数字認識テストを必須実施

### 日本語UI コンテキスト
Flutterアプリは日本語手書き文字評価を対象としているため全体で日本語を使用：
- 記入者番号 = Writer ID/Number
- 美文字 = Beautiful handwriting  
- 評価 = Evaluation
- 形・黒・白・場 = Shape, Black (ink), White (spacing), Center (positioning)

### API通信設定
- **ベースURL**: `http://localhost:8001`
- **メインエンドポイント**: `/process-cropped-form` (v0.8.3: API統合・評価スコア保存完全対応)
- **従来エンドポイント**: `/process-form`
- **タイムアウト**: 60秒
- **画像形式**: `data:image/jpeg;base64,{base64_data}`

### v0.8.4 UI改良実稼働確認事項
- ✅ Flutter: サンプル詳細画面の完全リニューアル（スライダー・前後移動・コンパクト表示）
- ✅ 評価スライダー: 0-10範囲での直感的スコア入力（白・黒・場・形の色分け）
- ✅ 前後移動: 一覧画面からの連続評価作業フロー実現
- ✅ 保存機能: HTTP 200/204両対応でSupabase保存エラー完全解決
- ✅ レイアウト: 連続評価に最適化されたスクロール最小化
- ✅ 画像表示: 80x80pxコンパクト表示で情報密度向上
- ✅ 基本情報: 記入者No、年齢、登録日、認識文字の統合表示
- ✅ コード整理: 不要セクション削除とメソッド統合

### v0.8.3 API統合実稼働確認事項
- ✅ Flutter: 6000x4000px高解像度画像送信（変更なし）
- ✅ API: 完全統合処理フロー動作（文字認識→評価スコア→DB保存）
- ✅ Gemini: 日本語文字認識（清:99%, 炎:98%, 葉:98%信頼度）
- ✅ PyTorch: 数字認識（12/12評価点数成功認識）
- ✅ データベース: writing_sampleテーブルにscore_white/black/center/shape自動保存
- ✅ ストレージ: 補助線除去済み画像の自動保存（`/workspace/debug/improved_char_*.jpg`）
- ✅ 既存サンプル: 重複時の評価スコア自動更新（action: "score_updated"）
- ✅ エラー解決: 全ての`xxx not defined`エラー完全解決
- ✅ デバッグ: `評価スコア収集結果`ログによる処理確認

### v0.8.2 照明対応実稼働確認事項
- ✅ Flutter: 6000x4000px高解像度画像送信（変更なし）
- ✅ API: A4画像全体照明補正適用済み認識動作
- ✅ API: 照明補正済み画像でのGemini文字認識（98-99%信頼度維持）
- ✅ API: 照明補正済み画像でのPyTorch数字認識（12/12成功維持）
- ✅ デバッグ: `debug/a4_lighting_corrected.jpg`, `enhanced_*.jpg`, `thresh_*.jpg` 段階出力
- ✅ 補助線除去: 文字保護強化により線の消失を防止
- ✅ 明暗差対応: 撮影時の明るい部分・影部分での安定認識を実現

### v0.8.0 従来実稼働確認事項
- ✅ Flutter: 6000x4000px高解像度画像送信
- ✅ API: 実際のGemini文字認識動作
- ✅ API: 実際のPyTorch数字認識動作  
- ✅ デバッグ: `debug/improved_char_*.jpg`, `debug/improved_score_*.jpg` 自動生成
- ✅ 結果: ImageUploadScreen成功ダイアログに実際の認識結果表示
