"""
Supabase統合 OCRプロセッサ
- improved_ocr_processor.py と supabase_client.py を統合
- Gemini認識結果をSupabaseに自動保存
- 画像ファイルを自動アップロード
Created: 2025-01-06
"""

import os
import sys
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import json
import logging

# パス追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.supabase_client import SupabaseClient

# 既存モジュールインポート
try:
    from core.gemini_client import GeminiCharacterRecognizer
    from core.improved_ocr_processor import ImprovedOCRProcessor
except ImportError as e:
    logging.warning(f"Import warning: {e}")
    # フォールバック
    GeminiCharacterRecognizer = None
    ImprovedOCRProcessor = None

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseOCRProcessor:
    """
    Supabase統合OCRプロセッサ
    - Gemini文字認識
    - Tesseract数字認識
    - 自動データベース保存
    - 画像ファイル管理
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None, 
                 bucket_name: str = 'ml-data', debug_enabled: bool = False):
        """
        初期化
        
        Args:
            supabase_url: Supabase Project URL
            supabase_key: Supabase API Key
            bucket_name: Storage bucket名
            debug_enabled: デバッグモード
        """
        # Supabaseクライアント初期化
        self.supabase = SupabaseClient(supabase_url, supabase_key)
        self.supabase.bucket_name = bucket_name
        
        # OCRプロセッサ初期化
        self.debug_enabled = debug_enabled
        self.debug_dir = "debug" if debug_enabled else None
        
        # Geminiクライアント初期化
        try:
            if GeminiCharacterRecognizer:
                self.gemini_client = GeminiCharacterRecognizer()
                self.use_gemini = True
                logger.info("Gemini client initialized")
            else:
                self.use_gemini = False
                logger.warning("Gemini client not available")
        except Exception as e:
            self.use_gemini = False
            logger.warning(f"Gemini initialization failed: {e}")
        
        # 既存OCRプロセッサ初期化
        try:
            if ImprovedOCRProcessor:
                self.ocr_processor = ImprovedOCRProcessor(
                    use_gemini=False,  # 独自にGemini管理
                    debug_dir=self.debug_dir
                )
                logger.info("OCR processor initialized")
            else:
                self.ocr_processor = None
                logger.warning("OCR processor not available")
        except Exception as e:
            self.ocr_processor = None
            logger.warning(f"OCR processor initialization failed: {e}")
    
    def process_form_image(self, image_path: str, writer_number: str, 
                          writer_age: int = None, writer_grade: str = None,
                          auto_save: bool = True) -> Dict[str, Any]:
        """
        記入用紙画像の完全処理
        
        Args:
            image_path: 画像ファイルパス
            writer_number: 記入者番号
            writer_age: 記入者年齢
            writer_grade: 記入者学年
            auto_save: 自動保存フラグ
            
        Returns:
            処理結果辞書
        """
        logger.info(f"Processing form image: {image_path}")
        
        try:
            # 画像読み込み
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # OCR処理実行
            if self.ocr_processor:
                ocr_results = self.ocr_processor.process_form(image_path, debug=self.debug_enabled)
            else:
                # 簡易フォールバック処理
                ocr_results = self._fallback_processing(image)
            
            # 結果統合
            results = {
                "source_image": image_path,
                "writer_number": writer_number,
                "writer_age": writer_age,
                "writer_grade": writer_grade,
                "processing_timestamp": datetime.now().isoformat(),
                "character_results": [],
                "number_results": [],
                "supabase_saved": False
            }
            
            # 文字認識結果処理
            if "character_recognition" in ocr_results:
                char_results = self._process_character_results(
                    ocr_results["character_recognition"], 
                    writer_number, writer_age, writer_grade, auto_save
                )
                results["character_results"] = char_results
            
            # 数字認識結果処理
            if "writer_number" in ocr_results or "evaluations" in ocr_results:
                number_results = self._process_number_results(ocr_results)
                results["number_results"] = number_results
            
            # 統計情報
            if auto_save:
                stats = self.supabase.get_stats()
                results["database_stats"] = stats
                results["supabase_saved"] = True
            
            logger.info(f"Form processing completed: {len(results['character_results'])} characters processed")
            return results
            
        except Exception as e:
            logger.error(f"Error processing form image: {e}")
            return {
                "error": str(e),
                "source_image": image_path,
                "supabase_saved": False
            }
    
    def _process_character_results(self, char_recognition: Dict[str, Any], 
                                 writer_number: str, writer_age: int, writer_grade: str,
                                 auto_save: bool) -> List[Dict[str, Any]]:
        """
        文字認識結果処理
        
        Args:
            char_recognition: 文字認識結果
            writer_number: 記入者番号
            writer_age: 記入者年齢
            writer_grade: 記入者学年
            auto_save: 自動保存フラグ
            
        Returns:
            処理済み文字結果リスト
        """
        character_results = []
        
        for char_key, char_data in char_recognition.items():
            try:
                # 画像データ準備
                if "image" in char_data:
                    image_array = char_data["image"]
                    # numpy配列をJPEGバイト配列に変換
                    _, buffer = cv2.imencode('.jpg', image_array)
                    image_bytes = buffer.tobytes()
                else:
                    logger.warning(f"No image data for {char_key}")
                    continue
                
                # Gemini認識実行
                gemini_result = None
                if self.use_gemini and image_array is not None:
                    try:
                        gemini_result = self.gemini_client.recognize_japanese_character(image_array)
                        logger.info(f"Gemini recognition for {char_key}: {gemini_result.get('character', 'Unknown')}")
                    except Exception as e:
                        logger.warning(f"Gemini recognition failed for {char_key}: {e}")
                
                # 結果構築
                result = {
                    "char_key": char_key,
                    "bbox": char_data.get("bbox"),
                    "gemini_result": gemini_result,
                    "image_size": image_array.shape if image_array is not None else None,
                    "saved_to_supabase": False
                }
                
                # Supabase保存
                if auto_save and gemini_result and image_bytes:
                    try:
                        # 認識された文字を使用
                        recognized_char = gemini_result.get('character', f'unknown_{char_key}')
                        
                        # データベース保存（重複チェック有効）
                        sample_data = self.supabase.create_writing_sample(
                            writer_number=writer_number,
                            character=recognized_char,
                            image_data=image_bytes,
                            gemini_result=gemini_result,
                            writer_age=writer_age,
                            writer_grade=writer_grade,
                            allow_duplicates=False  # 重複防止
                        )
                        
                        result["sample_id"] = sample_data["id"]
                        result["image_path"] = sample_data.get("image_path")
                        result["saved_to_supabase"] = True
                        result["is_duplicate"] = sample_data.get("is_duplicate", False)
                        result["action"] = sample_data.get("action", "created")
                        
                        if sample_data.get("is_duplicate"):
                            logger.info(f"Character already exists: {recognized_char} (ID: {sample_data['id']}) - SKIPPED")
                        else:
                            logger.info(f"Character saved to Supabase: {recognized_char} (ID: {sample_data['id']}) - CREATED")
                        
                    except Exception as e:
                        logger.error(f"Error saving character to Supabase: {e}")
                        result["save_error"] = str(e)
                
                character_results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing character {char_key}: {e}")
                character_results.append({
                    "char_key": char_key,
                    "error": str(e),
                    "saved_to_supabase": False
                })
        
        return character_results
    
    def _process_number_results(self, ocr_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        数字認識結果処理
        
        Args:
            ocr_results: OCR結果
            
        Returns:
            数字認識結果リスト
        """
        number_results = []
        
        # 記入者番号
        if "writer_number" in ocr_results:
            number_results.append({
                "type": "writer_number",
                "recognized_text": ocr_results["writer_number"].get("text"),
                "confidence": ocr_results["writer_number"].get("confidence"),
                "method": ocr_results["writer_number"].get("method")
            })
        
        # 評価数字
        if "evaluations" in ocr_results:
            for eval_key, eval_data in ocr_results["evaluations"].items():
                number_results.append({
                    "type": "evaluation",
                    "field": eval_key,
                    "recognized_text": eval_data.get("text"),
                    "confidence": eval_data.get("confidence"),
                    "method": eval_data.get("method")
                })
        
        return number_results
    
    def _fallback_processing(self, image: np.ndarray) -> Dict[str, Any]:
        """
        フォールバック処理（OCRプロセッサが利用できない場合）
        
        Args:
            image: 入力画像
            
        Returns:
            簡易処理結果
        """
        logger.warning("Using fallback processing")
        
        # 簡易的な文字領域検出
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # 仮想的な文字領域（右列の上から3つ）
        char_regions = []
        for i in range(3):
            y1 = int(height * 0.2 + i * height * 0.25)
            y2 = int(y1 + height * 0.15)
            x1 = int(width * 0.7)
            x2 = int(width * 0.9)
            
            char_image = image[y1:y2, x1:x2]
            char_regions.append({
                "image": char_image,
                "bbox": (x1, y1, x2, y2)
            })
        
        return {
            "character_recognition": {
                f"char_{i+1}": region for i, region in enumerate(char_regions)
            },
            "correction_applied": False,
            "fallback_processing": True
        }
    
    def update_sample_scores(self, sample_id: int, scores: Dict[str, int], 
                           comments: Dict[str, str] = None, evaluator: str = None) -> Dict[str, Any]:
        """
        サンプルの評価スコア更新
        
        Args:
            sample_id: サンプルID
            scores: 評価スコア
            comments: コメント
            evaluator: 評価者
            
        Returns:
            更新結果
        """
        try:
            result = self.supabase.update_writing_sample_scores(
                sample_id, scores, comments, evaluator
            )
            logger.info(f"Sample scores updated: ID={sample_id}")
            return result
        except Exception as e:
            logger.error(f"Error updating sample scores: {e}")
            raise
    
    def get_writer_samples(self, writer_number: str) -> List[Dict[str, Any]]:
        """
        記入者のサンプル一覧取得
        
        Args:
            writer_number: 記入者番号
            
        Returns:
            サンプルリスト
        """
        return self.supabase.get_writing_samples_by_writer(writer_number)
    
    def get_ml_dataset(self, quality_status: str = 'approved') -> List[Dict[str, Any]]:
        """
        機械学習用データセット取得
        
        Args:
            quality_status: 品質ステータス
            
        Returns:
            データセット
        """
        return self.supabase.get_ml_dataset(quality_status)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        データベース統計取得
        
        Returns:
            統計情報
        """
        return self.supabase.get_stats()

# ===========================
# メイン実行関数
# ===========================

def main():
    """
    メイン実行関数
    """
    # Supabase設定（service_role key使用）
    processor = SupabaseOCRProcessor(
        supabase_url="https://ypobmpkecniyuawxukol.supabase.co",
        supabase_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlwb2JtcGtlY25peXVhd3h1a29sIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTc4MTM3MywiZXhwIjoyMDY3MzU3MzczfQ.pRtpQHyo0kxLDt83pnk7RROKI6Q4KgaIlEm54VVVufY",
        bucket_name="ml-data",
        debug_enabled=True
    )
    
    # テスト画像処理
    test_image = "docs/記入sample.JPG"
    if os.path.exists(test_image):
        results = processor.process_form_image(
            image_path=test_image,
            writer_number="writer_demo",
            writer_age=20,
            writer_grade="大学",
            auto_save=True
        )
        
        print("=== Processing Results ===")
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        
        # 統計情報表示
        stats = processor.get_database_stats()
        print("\n=== Database Stats ===")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
    else:
        logger.error(f"Test image not found: {test_image}")
        
        # 統計情報のみ表示
        stats = processor.get_database_stats()
        print("=== Database Stats ===")
        print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()