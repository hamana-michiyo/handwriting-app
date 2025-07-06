"""
Supabase Database Client for 手書き文字評価システム
Created: 2025-01-06
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from supabase import create_client, Client
from PIL import Image
import io
import base64
import uuid
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseClient:
    """
    Supabase統合クライアント
    - データベース操作
    - ストレージ操作
    - 手書き文字評価データ管理
    """
    
    def __init__(self, url: str = None, key: str = None):
        """
        Supabaseクライアント初期化
        
        Args:
            url: Supabase Project URL
            key: Supabase API Key
        """
        self.url = url or os.getenv('SUPABASE_URL')
        self.key = key or os.getenv('SUPABASE_KEY')
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and Key must be provided")
        
        self.supabase: Client = create_client(self.url, self.key)
        self.bucket_name = os.getenv('SUPABASE_BUCKET', 'ml-data')
        
        logger.info(f"Supabase client initialized for bucket: {self.bucket_name}")
    
    # ===========================
    # WRITERS テーブル操作
    # ===========================
    
    def create_writer(self, writer_number: str, age: int = None, grade: str = None) -> Dict[str, Any]:
        """
        記入者作成
        
        Args:
            writer_number: 記入者番号 (例: "writer_001")
            age: 年齢
            grade: 学年
            
        Returns:
            作成された記入者データ
        """
        try:
            data = {
                "writer_number": writer_number,
                "age": age,
                "grade": grade
            }
            
            result = self.supabase.table('writers').insert(data).execute()
            logger.info(f"Writer created: {writer_number}")
            return result.data[0]
        except Exception as e:
            logger.error(f"Error creating writer: {e}")
            raise
    
    def get_writer_by_number(self, writer_number: str) -> Optional[Dict[str, Any]]:
        """
        記入者番号で記入者取得
        
        Args:
            writer_number: 記入者番号
            
        Returns:
            記入者データ または None
        """
        try:
            result = self.supabase.table('writers').select('*').eq('writer_number', writer_number).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting writer: {e}")
            return None
    
    def get_or_create_writer(self, writer_number: str, age: int = None, grade: str = None) -> Dict[str, Any]:
        """
        記入者取得または作成
        
        Args:
            writer_number: 記入者番号
            age: 年齢
            grade: 学年
            
        Returns:
            記入者データ
        """
        writer = self.get_writer_by_number(writer_number)
        if writer:
            return writer
        return self.create_writer(writer_number, age, grade)
    
    # ===========================
    # CHARACTERS テーブル操作
    # ===========================
    
    def create_character(self, character: str, stroke_count: int = None, 
                        difficulty_level: int = None, category: str = None) -> Dict[str, Any]:
        """
        文字マスタ作成
        
        Args:
            character: 文字
            stroke_count: 画数
            difficulty_level: 難易度 (1-5)
            category: カテゴリー (hiragana/katakana/kanji)
            
        Returns:
            作成された文字データ
        """
        try:
            data = {
                "character": character,
                "stroke_count": stroke_count,
                "difficulty_level": difficulty_level,
                "category": category
            }
            
            result = self.supabase.table('characters').insert(data).execute()
            logger.info(f"Character created: {character}")
            return result.data[0]
        except Exception as e:
            logger.error(f"Error creating character: {e}")
            raise
    
    def get_character_by_char(self, character: str) -> Optional[Dict[str, Any]]:
        """
        文字で文字マスタ取得
        
        Args:
            character: 文字
            
        Returns:
            文字データ または None
        """
        try:
            result = self.supabase.table('characters').select('*').eq('character', character).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting character: {e}")
            return None
    
    def get_or_create_character(self, character: str, stroke_count: int = None, 
                               difficulty_level: int = None, category: str = None) -> Dict[str, Any]:
        """
        文字マスタ取得または作成
        
        Args:
            character: 文字
            stroke_count: 画数
            difficulty_level: 難易度
            category: カテゴリー
            
        Returns:
            文字データ
        """
        char_data = self.get_character_by_char(character)
        if char_data:
            return char_data
        return self.create_character(character, stroke_count, difficulty_level, category)
    
    # ===========================
    # STORAGE 操作
    # ===========================
    
    def upload_image(self, image_data: bytes, file_path: str) -> str:
        """
        画像アップロード
        
        Args:
            image_data: 画像バイナリデータ
            file_path: ストレージ内のファイルパス
            
        Returns:
            アップロードされたファイルの公開URL
        """
        try:
            # ファイルアップロード
            result = self.supabase.storage.from_(self.bucket_name).upload(
                file_path, 
                image_data,
                file_options={"content-type": "image/jpeg"}
            )
            
            # 公開URL取得
            public_url = self.supabase.storage.from_(self.bucket_name).get_public_url(file_path)
            
            logger.info(f"Image uploaded: {file_path}")
            return public_url
            
        except Exception as e:
            logger.error(f"Error uploading image: {e}")
            raise
    
    def generate_image_path(self, writer_number: str, character: str, sample_id: int = None) -> str:
        """
        画像パス生成（日本語文字対応）
        
        Args:
            writer_number: 記入者番号
            character: 文字
            sample_id: サンプルID (オプション)
            
        Returns:
            生成された画像パス
        """
        now = datetime.now()
        date_path = f"{now.year:04d}/{now.month:02d}/{now.day:02d}"
        
        # 日本語文字の代わりにUUIDを使用（ファイル名安全）
        if sample_id:
            filename = f"{sample_id}.jpg"
        else:
            filename = f"{uuid.uuid4().hex[:8]}.jpg"
        
        return f"writing-samples/{date_path}/{writer_number}/{filename}"
    
    # ===========================
    # WRITING_SAMPLES テーブル操作
    # ===========================
    
    def create_writing_sample(self, writer_number: str, character: str, 
                            image_data: bytes, scores: Dict[str, int] = None,
                            comments: Dict[str, str] = None, 
                            gemini_result: Dict[str, Any] = None,
                            writer_age: int = None, writer_grade: str = None,
                            allow_duplicates: bool = True) -> Dict[str, Any]:
        """
        手書きサンプル作成
        
        Args:
            writer_number: 記入者番号
            character: 文字
            image_data: 画像データ
            scores: 評価スコア辞書 {'white': 8, 'black': 7, ...}
            comments: コメント辞書 {'white': 'Good!', ...}
            gemini_result: Gemini認識結果
            writer_age: 記入者年齢
            writer_grade: 記入者学年
            allow_duplicates: 重複許可フラグ
            
        Returns:
            作成された手書きサンプルデータ または 既存データ
        """
        try:
            # 記入者取得・作成
            writer = self.get_or_create_writer(writer_number, writer_age, writer_grade)
            
            # 文字マスタ取得・作成
            char_data = self.get_or_create_character(character)
            
            # 重複チェック
            if not allow_duplicates:
                existing_sample = self._check_existing_sample(writer['id'], char_data['id'])
                if existing_sample:
                    logger.info(f"Existing sample found: Writer={writer_number}, Character={character}, ID={existing_sample['id']}")
                    return {
                        **existing_sample,
                        'is_duplicate': True,
                        'action': 'skipped'
                    }
            
            # 画像パス生成
            image_path = self.generate_image_path(writer_number, character)
            image_filename = image_path.split('/')[-1]
            
            # 画像アップロード
            public_url = self.upload_image(image_data, image_path)
            
            # データベース挿入
            sample_data = {
                "writer_id": writer['id'],
                "character_id": char_data['id'],
                "image_filename": image_filename,
                "image_path": image_path,
            }
            
            # スコア設定
            if scores:
                sample_data.update({
                    "score_white": scores.get('white'),
                    "score_black": scores.get('black'),
                    "score_center": scores.get('center'),
                    "score_shape": scores.get('shape'),
                })
            
            # コメント設定
            if comments:
                sample_data.update({
                    "comment_white": comments.get('white'),
                    "comment_black": comments.get('black'),
                    "comment_center": comments.get('center'),
                    "comment_shape": comments.get('shape'),
                })
            
            # Gemini認識結果設定
            if gemini_result:
                sample_data.update({
                    "gemini_recognized_char": gemini_result.get('character'),
                    "gemini_confidence": gemini_result.get('confidence'),
                    "gemini_alternatives": gemini_result.get('alternatives'),
                    "gemini_reasoning": gemini_result.get('reasoning'),
                })
            
            result = self.supabase.table('writing_samples').insert(sample_data).execute()
            sample = result.data[0]
            
            logger.info(f"Writing sample created: ID={sample['id']}, Writer={writer_number}, Character={character}")
            return sample
            
        except Exception as e:
            logger.error(f"Error creating writing sample: {e}")
            raise
    
    def _check_existing_sample(self, writer_id: int, character_id: int) -> Optional[Dict[str, Any]]:
        """
        既存サンプルチェック
        
        Args:
            writer_id: 記入者ID
            character_id: 文字ID
            
        Returns:
            既存サンプル または None
        """
        try:
            result = self.supabase.table('writing_samples').select('*').eq('writer_id', writer_id).eq('character_id', character_id).limit(1).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error checking existing sample: {e}")
            return None
    
    def update_writing_sample_scores(self, sample_id: int, scores: Dict[str, int], 
                                   comments: Dict[str, str] = None, 
                                   evaluated_by: str = None) -> Dict[str, Any]:
        """
        手書きサンプルの評価更新
        
        Args:
            sample_id: サンプルID
            scores: 評価スコア辞書
            comments: コメント辞書
            evaluated_by: 評価者
            
        Returns:
            更新された手書きサンプルデータ
        """
        try:
            update_data = {
                "score_white": scores.get('white'),
                "score_black": scores.get('black'),
                "score_center": scores.get('center'),
                "score_shape": scores.get('shape'),
                "evaluated_by": evaluated_by,
            }
            
            if comments:
                update_data.update({
                    "comment_white": comments.get('white'),
                    "comment_black": comments.get('black'),
                    "comment_center": comments.get('center'),
                    "comment_shape": comments.get('shape'),
                })
            
            result = self.supabase.table('writing_samples').update(update_data).eq('id', sample_id).execute()
            
            logger.info(f"Writing sample updated: ID={sample_id}")
            return result.data[0]
            
        except Exception as e:
            logger.error(f"Error updating writing sample: {e}")
            raise
    
    def get_writing_samples_by_writer(self, writer_number: str) -> List[Dict[str, Any]]:
        """
        記入者別手書きサンプル取得
        
        Args:
            writer_number: 記入者番号
            
        Returns:
            手書きサンプルリスト
        """
        try:
            result = self.supabase.table('writing_samples').select(
                """
                id, image_filename, image_path, 
                score_white, score_black, score_center, score_shape, score_overall,
                comment_white, comment_black, comment_center, comment_shape,
                gemini_recognized_char, gemini_confidence,
                quality_status, created_at,
                writers(writer_number, age, grade),
                characters(character, stroke_count, difficulty_level, category)
                """
            ).eq('writers.writer_number', writer_number).execute()
            
            return result.data
            
        except Exception as e:
            logger.error(f"Error getting writing samples: {e}")
            return []
    
    def get_ml_dataset(self, quality_status: str = 'approved') -> List[Dict[str, Any]]:
        """
        機械学習用データセット取得
        
        Args:
            quality_status: 品質ステータス
            
        Returns:
            機械学習用データセット
        """
        try:
            result = self.supabase.table('ml_dataset').select('*').eq('quality_status', quality_status).execute()
            return result.data
            
        except Exception as e:
            logger.error(f"Error getting ML dataset: {e}")
            return []
    
    # ===========================
    # ユーティリティ関数
    # ===========================
    
    def image_to_bytes(self, image_path: str) -> bytes:
        """
        画像ファイルをバイト配列に変換
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            画像バイナリデータ
        """
        with open(image_path, 'rb') as f:
            return f.read()
    
    def pil_image_to_bytes(self, pil_image: Image.Image, format: str = 'JPEG') -> bytes:
        """
        PIL画像をバイト配列に変換
        
        Args:
            pil_image: PIL画像オブジェクト
            format: 画像フォーマット
            
        Returns:
            画像バイナリデータ
        """
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format=format)
        return img_buffer.getvalue()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        統計情報取得
        
        Returns:
            統計情報辞書
        """
        try:
            # 基本統計
            writers_count = self.supabase.table('writers').select('id', count='exact').execute().count
            characters_count = self.supabase.table('characters').select('id', count='exact').execute().count
            samples_count = self.supabase.table('writing_samples').select('id', count='exact').execute().count
            
            # 品質別統計
            approved_count = self.supabase.table('writing_samples').select('id', count='exact').eq('quality_status', 'approved').execute().count
            pending_count = self.supabase.table('writing_samples').select('id', count='exact').eq('quality_status', 'pending').execute().count
            
            return {
                'writers_count': writers_count,
                'characters_count': characters_count,
                'samples_count': samples_count,
                'approved_samples': approved_count,
                'pending_samples': pending_count,
                'approval_rate': (approved_count / samples_count * 100) if samples_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

# ===========================
# 使用例・テスト用関数
# ===========================

def test_supabase_client():
    """
    Supabaseクライアントテスト
    """
    # 環境変数設定
    client = SupabaseClient(
        url="https://ypobmpkecniyuawxukol.supabase.co",
        key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlwb2JtcGtlY25peXVhd3h1a29sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE3ODEzNzMsImV4cCI6MjA2NzM1NzM3M30.JdrURiuZJ4HvFo32bUTfr3ELLRS8BzFhBBldapvzGjw"
    )
    
    # 統計情報取得
    stats = client.get_stats()
    print("Database Stats:", stats)
    
    # 記入者取得
    writer = client.get_or_create_writer("writer_test", 10, "小4")
    print("Writer:", writer)
    
    # 文字マスタ取得
    char = client.get_or_create_character("テ", 3, 1, "katakana")
    print("Character:", char)

if __name__ == "__main__":
    test_supabase_client()