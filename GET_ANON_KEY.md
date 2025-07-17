# Supabase Anonymous Key取得手順

現在、Flutterアプリ用のAnonymous Keyが必要です。

## 取得手順

1. [Supabase Dashboard](https://supabase.com/dashboard)にアクセス
2. プロジェクト `ypobmpkecniyuawxukol` を選択
3. 左側メニューの「Settings」→「API」をクリック
4. 「Project API keys」セクションで以下を確認:
   - **anon public key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` (Flutter用)
   - **service_role secret**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` (API用・設定済み)

## 設定が必要なファイル

- `lib/services/supabase_service.dart:10` - _supabaseAnonKey
  
## セキュリティ注意

- **anon key**: 公開可能（クライアントアプリ用）
- **service_role key**: 秘匿必須（サーバー用・機密情報）

匿名キーを取得後、`supabase_service.dart`に設定してください。