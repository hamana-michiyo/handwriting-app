# Supabase設定手順

## 概要
このドキュメントは、Flutter一覧管理画面を実際のSupabaseデータベースに接続するための設定手順を説明します。

## 1. Supabaseプロジェクトの作成

### 1.1 Supabaseアカウント作成
1. [Supabase](https://supabase.com)にアクセス
2. 「Start your project」をクリック
3. GitHub/Google/Email認証でアカウント作成

### 1.2 新しいプロジェクト作成
1. ダッシュボードで「New project」をクリック
2. Organization選択
3. プロジェクト名: `handwriting-evaluation`
4. データベースパスワード設定
5. Region: `Northeast Asia (Tokyo)` または `Southeast Asia (Singapore)`
6. 「Create new project」をクリック

## 2. データベースセットアップ

### 2.1 writing_sampleテーブル作成
Supabase SQL Editorで以下のSQLを実行:

```sql
-- writing_sampleテーブル作成
CREATE TABLE writing_sample (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    writer_number TEXT NOT NULL,
    writer_age INTEGER,
    writer_grade TEXT,
    image_path TEXT,
    original_image_path TEXT,
    character_results JSONB,
    number_results JSONB,
    evaluations JSONB,
    score_white INTEGER,
    score_black INTEGER,
    score_center INTEGER,
    score_shape INTEGER,
    gemini_recognition JSONB,
    perspective_corrected BOOLEAN DEFAULT FALSE,
    corner_detection_success BOOLEAN DEFAULT FALSE,
    processing_time_seconds DECIMAL(10,3)
);

-- インデックス作成
CREATE INDEX idx_writing_sample_writer_number ON writing_sample(writer_number);
CREATE INDEX idx_writing_sample_created_at ON writing_sample(created_at);
CREATE INDEX idx_writing_sample_scores ON writing_sample(score_white, score_black, score_center, score_shape);

-- RLS（Row Level Security）を有効化
ALTER TABLE writing_sample ENABLE ROW LEVEL SECURITY;

-- 全ユーザーに読み取り権限を付与（必要に応じて制限）
CREATE POLICY "Allow read access to all users" ON writing_sample
    FOR SELECT USING (true);

-- 全ユーザーに書き込み権限を付与（必要に応じて制限）
CREATE POLICY "Allow insert access to all users" ON writing_sample
    FOR INSERT WITH CHECK (true);

-- 全ユーザーに更新権限を付与（必要に応じて制限）
CREATE POLICY "Allow update access to all users" ON writing_sample
    FOR UPDATE USING (true);

-- 全ユーザーに削除権限を付与（必要に応じて制限）
CREATE POLICY "Allow delete access to all users" ON writing_sample
    FOR DELETE USING (true);
```

### 2.2 ストレージバケット作成
1. Supabase Dashboard → Storage
2. 「Create bucket」をクリック
3. Bucket名: `handwriting-images`
4. Public: `true`（必要に応じて制限）
5. 「Create bucket」をクリック

### 2.3 ストレージポリシー設定
```sql
-- ストレージバケットへの読み取り権限
CREATE POLICY "Allow read access to handwriting images" ON storage.objects
    FOR SELECT USING (bucket_id = 'handwriting-images');

-- ストレージバケットへの書き込み権限
CREATE POLICY "Allow insert access to handwriting images" ON storage.objects
    FOR INSERT WITH CHECK (bucket_id = 'handwriting-images');

-- ストレージバケットへの更新権限
CREATE POLICY "Allow update access to handwriting images" ON storage.objects
    FOR UPDATE USING (bucket_id = 'handwriting-images');

-- ストレージバケットへの削除権限
CREATE POLICY "Allow delete access to handwriting images" ON storage.objects
    FOR DELETE USING (bucket_id = 'handwriting-images');
```

## 3. API認証情報の取得

### 3.1 プロジェクト設定確認
1. Supabase Dashboard → Settings → API
2. 以下の情報を確認:
   - **Project URL**: `https://your-project-id.supabase.co`
   - **anon public key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
   - **service_role secret**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

### 3.2 認証情報のメモ
- Project URL: `https://your-project-id.supabase.co`
- Anon Key: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

## 4. Flutter アプリケーション設定

### 4.1 SupabaseService設定更新
`lib/services/supabase_service.dart`で以下を更新:

```dart
class SupabaseService {
  // 実際のSupabase設定に変更
  static const String _supabaseUrl = 'https://your-project-id.supabase.co';
  static const String _supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...';
  
  // 以下は変更不要
  static const Duration _timeout = Duration(seconds: 30);
  static const String _writingSampleTable = 'writing_sample';
```

### 4.2 モックモードの無効化
`lib/screens/sample_list_screen.dart`で以下を変更:

```dart
class _SampleListScreenState extends State<SampleListScreen> {
  // 実際のSupabaseを使用する場合はfalseに設定
  static const bool _useMockMode = false;  // ← trueからfalseに変更
```

### 4.3 環境変数での管理（推奨）
プロダクション環境では環境変数を使用することを推奨:

```dart
class SupabaseService {
  static const String _supabaseUrl = String.fromEnvironment(
    'SUPABASE_URL',
    defaultValue: 'https://your-project-id.supabase.co',
  );
  
  static const String _supabaseAnonKey = String.fromEnvironment(
    'SUPABASE_ANON_KEY',
    defaultValue: 'your-anon-key',
  );
```

## 5. 動作確認

### 5.1 接続テスト
1. Flutterアプリを起動
2. 一覧管理画面に移動
3. 統計情報が正しく表示されることを確認
4. データが空の場合は「該当するサンプルがありません」と表示

### 5.2 データ投入テスト
1. Python API (`handwriting-eval-api`) を起動
2. Flutter アプリから撮影→アップロード
3. 一覧画面で新しいデータが表示されることを確認

## 6. トラブルシューティング

### 6.1 接続エラーの場合
- Project URLが正しいか確認
- Anon Keyが正しいか確認
- ネットワーク接続を確認

### 6.2 権限エラーの場合
- RLSポリシーが正しく設定されているか確認
- Anon Keyに適切な権限があるか確認

### 6.3 データが表示されない場合
- テーブル名が正しいか確認（`writing_sample`）
- データが実際に存在するか確認
- JSONBフィールドの構造が正しいか確認

## 7. セキュリティ考慮事項

### 7.1 本番環境での設定
- RLSポリシーを適切に制限
- Service Roleキーは使用しない
- HTTPS通信を強制
- 認証情報を環境変数で管理

### 7.2 API制限
- レート制限の設定
- 不正なリクエストの監視
- ログの監視とアラート設定

## 8. 関連ファイル

- `lib/services/supabase_service.dart` - Supabase通信サービス
- `lib/services/mock_supabase_service.dart` - モックサービス（開発用）
- `lib/screens/sample_list_screen.dart` - 一覧管理画面
- `handwriting-eval-api/supabase_schema.sql` - データベーススキーマ
- `handwriting-eval-api/src/database/supabase_client.py` - Python側Supabase連携

## 9. 参考リンク

- [Supabase公式ドキュメント](https://supabase.com/docs)
- [Flutter Supabase クライアント](https://pub.dev/packages/supabase_flutter)
- [Row Level Security](https://supabase.com/docs/guides/auth/row-level-security)
- [Supabase Storage](https://supabase.com/docs/guides/storage)