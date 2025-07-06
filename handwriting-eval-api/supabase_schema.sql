-- =====================================================
-- Supabase Database Schema for 手書き文字評価システム
-- Created: 2025-01-06
-- =====================================================

-- 既存テーブル削除（必要に応じて）
-- DROP TABLE IF EXISTS evaluation_history CASCADE;
-- DROP TABLE IF EXISTS writing_samples CASCADE;
-- DROP TABLE IF EXISTS evaluation_sessions CASCADE;
-- DROP TABLE IF EXISTS reference_images CASCADE;
-- DROP TABLE IF EXISTS characters CASCADE;
-- DROP TABLE IF EXISTS writers CASCADE;

-- =====================================================
-- 1. WRITERS テーブル (記入者情報 - 匿名対応)
-- =====================================================
CREATE TABLE writers (
    id SERIAL PRIMARY KEY,
    writer_number VARCHAR(20) UNIQUE NOT NULL,  -- "writer_001", "writer_002"等
    age INTEGER CHECK (age >= 3 AND age <= 100),
    grade VARCHAR(20),  -- "幼稚園", "小1", "小2", "中1", "高1", "大学", "社会人"
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- 2. CHARACTERS テーブル (文字マスタ)
-- =====================================================
CREATE TABLE characters (
    id SERIAL PRIMARY KEY,
    character VARCHAR(10) UNIQUE NOT NULL,
    stroke_count INTEGER,
    difficulty_level INTEGER CHECK (difficulty_level >= 1 AND difficulty_level <= 5),
    category VARCHAR(20) CHECK (category IN ('hiragana', 'katakana', 'kanji')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- 3. WRITING_SAMPLES テーブル (手書きサンプル - メインテーブル)
-- =====================================================
CREATE TABLE writing_samples (
    id SERIAL PRIMARY KEY,
    writer_id INTEGER REFERENCES writers(id) ON DELETE CASCADE,
    character_id INTEGER REFERENCES characters(id) ON DELETE CASCADE,
    
    -- 画像ファイル情報
    image_filename VARCHAR(255) NOT NULL,
    image_path VARCHAR(500) NOT NULL,
    
    -- 評価スコア (0-10)
    score_white INTEGER CHECK (score_white >= 0 AND score_white <= 10),
    score_black INTEGER CHECK (score_black >= 0 AND score_black <= 10),
    score_center INTEGER CHECK (score_center >= 0 AND score_center <= 10),
    score_shape INTEGER CHECK (score_shape >= 0 AND score_shape <= 10),
    score_overall DECIMAL(3,1) CHECK (score_overall >= 0 AND score_overall <= 10),
    
    -- 評価コメント
    comment_white TEXT,
    comment_black TEXT,
    comment_center TEXT,
    comment_shape TEXT,
    comment_overall TEXT,
    
    -- AI認識情報
    gemini_recognized_char VARCHAR(10),
    gemini_confidence DECIMAL(5,2),
    gemini_alternatives JSONB,
    gemini_reasoning TEXT,
    
    -- メタデータ
    is_reference BOOLEAN DEFAULT FALSE,
    quality_status VARCHAR(20) DEFAULT 'pending' CHECK (quality_status IN ('pending', 'approved', 'rejected')),
    
    -- 監査フィールド
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    evaluated_by VARCHAR(50),
    evaluated_at TIMESTAMP WITH TIME ZONE
);

-- =====================================================
-- 4. EVALUATION_SESSIONS テーブル (評価セッション)
-- =====================================================
CREATE TABLE evaluation_sessions (
    id SERIAL PRIMARY KEY,
    writer_id INTEGER REFERENCES writers(id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    total_characters INTEGER DEFAULT 0,
    average_score DECIMAL(3,1),
    device_info JSONB,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- 5. REFERENCE_IMAGES テーブル (参考画像)
-- =====================================================
CREATE TABLE reference_images (
    id SERIAL PRIMARY KEY,
    character_id INTEGER REFERENCES characters(id) ON DELETE CASCADE,
    image_filename VARCHAR(255) NOT NULL,
    image_path VARCHAR(500) NOT NULL,
    font_type VARCHAR(50),
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- 6. EVALUATION_HISTORY テーブル (評価履歴)
-- =====================================================
CREATE TABLE evaluation_history (
    id SERIAL PRIMARY KEY,
    writing_sample_id INTEGER REFERENCES writing_samples(id) ON DELETE CASCADE,
    
    -- 変更前の値 (0-10)
    old_score_white INTEGER,
    old_score_black INTEGER,
    old_score_center INTEGER,
    old_score_shape INTEGER,
    old_comment_white TEXT,
    old_comment_black TEXT,
    old_comment_center TEXT,
    old_comment_shape TEXT,
    
    -- 変更後の値 (0-10)
    new_score_white INTEGER,
    new_score_black INTEGER,
    new_score_center INTEGER,
    new_score_shape INTEGER,
    new_comment_white TEXT,
    new_comment_black TEXT,
    new_comment_center TEXT,
    new_comment_shape TEXT,
    
    -- 変更情報
    changed_by VARCHAR(50),
    change_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- インデックス作成
-- =====================================================

-- 基本インデックス
CREATE INDEX idx_writers_number ON writers(writer_number);
CREATE INDEX idx_characters_char ON characters(character);
CREATE INDEX idx_writing_samples_writer ON writing_samples(writer_id);
CREATE INDEX idx_writing_samples_character ON writing_samples(character_id);
CREATE INDEX idx_writing_samples_quality ON writing_samples(quality_status);
CREATE INDEX idx_writing_samples_created ON writing_samples(created_at DESC);

-- 機械学習用複合インデックス
CREATE INDEX idx_ml_dataset ON writing_samples(character_id, writer_id, quality_status) 
WHERE quality_status = 'approved';

-- 年齢別分析用インデックス
CREATE INDEX idx_writer_age_analysis ON writers(age, grade);

-- 評価履歴用インデックス
CREATE INDEX idx_evaluation_history_sample ON evaluation_history(writing_sample_id);
CREATE INDEX idx_evaluation_history_created ON evaluation_history(created_at DESC);

-- =====================================================
-- 自動更新トリガー関数
-- =====================================================

-- writers テーブルの updated_at 自動更新
CREATE OR REPLACE FUNCTION update_writers_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER writers_updated_at_trigger
    BEFORE UPDATE ON writers
    FOR EACH ROW
    EXECUTE FUNCTION update_writers_updated_at();

-- writing_samples テーブルの更新トリガー
CREATE OR REPLACE FUNCTION update_writing_samples_trigger()
RETURNS TRIGGER AS $$
BEGIN
    -- 評価が変更された場合、履歴に記録
    IF OLD.score_white IS DISTINCT FROM NEW.score_white OR 
       OLD.score_black IS DISTINCT FROM NEW.score_black OR 
       OLD.score_center IS DISTINCT FROM NEW.score_center OR 
       OLD.score_shape IS DISTINCT FROM NEW.score_shape OR
       OLD.comment_white IS DISTINCT FROM NEW.comment_white OR
       OLD.comment_black IS DISTINCT FROM NEW.comment_black OR
       OLD.comment_center IS DISTINCT FROM NEW.comment_center OR
       OLD.comment_shape IS DISTINCT FROM NEW.comment_shape THEN
        
        INSERT INTO evaluation_history (
            writing_sample_id,
            old_score_white, old_score_black, old_score_center, old_score_shape,
            old_comment_white, old_comment_black, old_comment_center, old_comment_shape,
            new_score_white, new_score_black, new_score_center, new_score_shape,
            new_comment_white, new_comment_black, new_comment_center, new_comment_shape,
            changed_by
        ) VALUES (
            NEW.id,
            OLD.score_white, OLD.score_black, OLD.score_center, OLD.score_shape,
            OLD.comment_white, OLD.comment_black, OLD.comment_center, OLD.comment_shape,
            NEW.score_white, NEW.score_black, NEW.score_center, NEW.score_shape,
            NEW.comment_white, NEW.comment_black, NEW.comment_center, NEW.comment_shape,
            NEW.evaluated_by
        );
    END IF;
    
    -- 総合スコア自動計算 (各項目が入力されている場合)
    IF NEW.score_white IS NOT NULL AND NEW.score_black IS NOT NULL AND 
       NEW.score_center IS NOT NULL AND NEW.score_shape IS NOT NULL THEN
        NEW.score_overall = (
            NEW.score_white * 0.3 +
            NEW.score_black * 0.2 +
            NEW.score_center * 0.2 +
            NEW.score_shape * 0.3
        );
    END IF;
    
    -- 評価日時自動設定
    IF NEW.score_overall IS NOT NULL AND OLD.score_overall IS NULL THEN
        NEW.evaluated_at = NOW();
    END IF;
    
    NEW.updated_at = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER writing_samples_update_trigger
    BEFORE UPDATE ON writing_samples
    FOR EACH ROW
    EXECUTE FUNCTION update_writing_samples_trigger();

-- =====================================================
-- Row Level Security (RLS) 設定
-- =====================================================

-- writing_samples テーブルのRLS有効化
ALTER TABLE writing_samples ENABLE ROW LEVEL SECURITY;

-- 管理者権限での全データアクセス許可
CREATE POLICY "Allow admin access" ON writing_samples
FOR ALL TO authenticated
USING (auth.jwt() ->> 'role' = 'admin');

-- 一般ユーザーは承認済みデータのみ閲覧可能
CREATE POLICY "Allow approved data read" ON writing_samples
FOR SELECT TO authenticated
USING (quality_status = 'approved');

-- 匿名ユーザーは承認済みデータのみ閲覧可能
CREATE POLICY "Allow public read approved" ON writing_samples
FOR SELECT TO anon
USING (quality_status = 'approved');

-- その他のテーブルも必要に応じてRLS設定
-- ALTER TABLE writers ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE characters ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE evaluation_history ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 便利な関数・ビュー
-- =====================================================

-- 機械学習用データ取得ビュー
CREATE VIEW ml_dataset AS
SELECT 
    ws.id,
    w.writer_number,
    w.age,
    w.grade,
    c.character,
    c.stroke_count,
    c.difficulty_level,
    c.category,
    ws.image_path,
    ws.score_white,
    ws.score_black,
    ws.score_center,
    ws.score_shape,
    ws.score_overall,
    ws.gemini_recognized_char,
    ws.gemini_confidence,
    ws.created_at
FROM writing_samples ws
JOIN writers w ON ws.writer_id = w.id
JOIN characters c ON ws.character_id = c.id
WHERE ws.quality_status = 'approved';

-- 評価統計ビュー
CREATE VIEW evaluation_stats AS
SELECT 
    c.character,
    COUNT(*) as total_samples,
    AVG(ws.score_overall) as avg_score,
    MIN(ws.score_overall) as min_score,
    MAX(ws.score_overall) as max_score,
    AVG(w.age) as avg_writer_age
FROM writing_samples ws
JOIN writers w ON ws.writer_id = w.id
JOIN characters c ON ws.character_id = c.id
WHERE ws.quality_status = 'approved'
GROUP BY c.character
ORDER BY c.character;

-- 記入者別統計ビュー
CREATE VIEW writer_stats AS
SELECT 
    w.writer_number,
    w.age,
    w.grade,
    COUNT(*) as total_samples,
    AVG(ws.score_overall) as avg_score,
    COUNT(DISTINCT ws.character_id) as unique_characters
FROM writers w
JOIN writing_samples ws ON w.id = ws.writer_id
WHERE ws.quality_status = 'approved'
GROUP BY w.id, w.writer_number, w.age, w.grade
ORDER BY w.writer_number;

-- =====================================================
-- サンプルデータ挿入（オプション）
-- =====================================================

-- 基本文字データ
INSERT INTO characters (character, stroke_count, difficulty_level, category) VALUES 
('清', 11, 4, 'kanji'),
('炎', 8, 3, 'kanji'),
('葉', 12, 4, 'kanji'),
('あ', 3, 1, 'hiragana'),
('か', 3, 1, 'hiragana'),
('さ', 3, 1, 'hiragana'),
('た', 4, 1, 'hiragana'),
('な', 4, 1, 'hiragana'),
('は', 3, 1, 'hiragana'),
('ま', 3, 1, 'hiragana'),
('や', 3, 1, 'hiragana'),
('ら', 2, 1, 'hiragana'),
('わ', 3, 1, 'hiragana'),
('を', 3, 1, 'hiragana'),
('ん', 1, 1, 'hiragana');

-- サンプル記入者
INSERT INTO writers (writer_number, age, grade) VALUES 
('writer_001', 8, '小2'),
('writer_002', 15, '中3'),
('writer_003', 22, '大学');

-- =====================================================
-- 完了メッセージ
-- =====================================================
-- データベーススキーマの作成が完了しました。
-- 
-- 次の手順:
-- 1. Supabase Dashboard で Storage バケット 'writing-samples' を作成
-- 2. 必要に応じてRLSポリシーを調整
-- 3. API経由でのデータ挿入テスト
-- 4. Flutter/Web アプリからの接続テスト
-- 
-- ファイル管理構造:
-- writing-samples/YYYY/MM/DD/writer_XXX/ID_文字.jpg
-- reference/ID_文字_primary.jpg
-- =====================================================