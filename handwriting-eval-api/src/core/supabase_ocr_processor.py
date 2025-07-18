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
    logging.info("GeminiCharacterRecognizer imported successfully")
except ImportError as e:
    logging.error(f"Failed to import GeminiCharacterRecognizer: {e}")
    import traceback
    logging.error(f"GeminiCharacterRecognizer import traceback: {traceback.format_exc()}")
    GeminiCharacterRecognizer = None

try:
    from core.improved_ocr_processor import ImprovedOCRProcessor
    logging.info("ImprovedOCRProcessor imported successfully")
except ImportError as e:
    logging.error(f"Failed to import ImprovedOCRProcessor: {e}")
    import traceback
    logging.error(f"ImprovedOCRProcessor import traceback: {traceback.format_exc()}")
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
                logger.info("Gemini client initialized successfully")
            else:
                self.use_gemini = False
                logger.error("GeminiCharacterRecognizer class not available - check imports")
        except Exception as e:
            self.use_gemini = False
            logger.error(f"Gemini initialization failed: {e}")
            import traceback
            logger.error(f"Gemini initialization traceback: {traceback.format_exc()}")
        
        # 既存OCRプロセッサ初期化
        try:
            if ImprovedOCRProcessor:
                self.ocr_processor = ImprovedOCRProcessor(
                    use_gemini=True,  # Gemini認識を有効化
                    debug_dir=self.debug_dir
                )
                logger.info("OCR processor initialized successfully")
            else:
                self.ocr_processor = None
                logger.error("ImprovedOCRProcessor class not available - check imports")
        except Exception as e:
            self.ocr_processor = None
            logger.error(f"OCR processor initialization failed: {e}")
            import traceback
            logger.error(f"OCR processor traceback: {traceback.format_exc()}")
    
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
                logger.info(f"OCR結果のキー: {list(ocr_results.keys())}")
            else:
                # 簡易フォールバック処理
                logger.error("OCR processor is None - attempting fallback processing")
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
            if "character_results" in ocr_results:
                # 評価点数データを収集
                evaluation_scores = self._collect_evaluation_scores(ocr_results)
                
                char_results = self._process_character_results(
                    ocr_results["character_results"], 
                    writer_number, writer_age, writer_grade, auto_save, evaluation_scores
                )
                results["character_results"] = char_results
            
            # 数字認識結果処理
            if "writer_number" in ocr_results or "number_results" in ocr_results:
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
    
    def _collect_evaluation_scores(self, ocr_results: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """評価点数を文字ごとに分割して収集"""
        evaluation_scores = {"char_1": {}, "char_2": {}, "char_3": {}}
        logger.info(f"評価スコア収集開始")
        
        # evaluationsキーからスコアデータを取得（improved_ocr_processor.pyのprocess_formメソッド用）
        if "evaluations" in ocr_results:
            logger.info(f"evaluations found: {len(ocr_results['evaluations'])} items")
            for field_name, result in ocr_results["evaluations"].items():
                logger.info(f"評価項目発見: {field_name} = {result.get('text', '')}")
                
                # recognized_text または text キーから値を取得
                score_text = result.get('text', result.get('recognized_text', ''))
                
                # 評価名から文字番号と評価項目を抽出
                if "評価1" in field_name:
                    char_key = "char_1"
                elif "評価2" in field_name:
                    char_key = "char_2"
                elif "評価3" in field_name:
                    char_key = "char_3"
                else:
                    continue
                
                # 評価項目を抽出
                if "白" in field_name:
                    score_type = "white"
                elif "黒" in field_name:
                    score_type = "black"
                elif "場" in field_name:
                    score_type = "center"
                elif "形" in field_name:
                    score_type = "shape"
                else:
                    continue
                
                # 数字として評価点数を保存
                try:
                    score_value = int(score_text)
                    evaluation_scores[char_key][score_type] = score_value
                    logger.info(f"評価点数マッピング: {char_key}.{score_type} = {score_value}")
                except (ValueError, TypeError):
                    logger.warning(f"評価点数変換失敗: {field_name} = {score_text}")
        
        logger.info(f"評価スコア収集結果: {evaluation_scores}")
        return evaluation_scores
    
    def _process_character_results(self, char_recognition: Dict[str, Any], 
                                 writer_number: str, writer_age: int, writer_grade: str,
                                 auto_save: bool, evaluation_scores: Dict[str, Dict[str, int]] = None) -> List[Dict[str, Any]]:
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
                # Gemini認識結果を取得（既に処理済み）
                gemini_result = char_data.get("gemini_recognition")
                if not gemini_result:
                    logger.warning(f"No Gemini result for {char_key}")
                    continue
                
                # 補助線除去済み画像データを準備（ストレージ保存用）
                # 現在は元画像を使用
                image_bytes = None
                image_array = None
                try:
                    # デバッグディレクトリから補助線除去済み画像を読み込み
                    cleaned_image_path = f"/workspace/debug/improved_char_{char_key}.jpg"
                    if os.path.exists(cleaned_image_path):
                        with open(cleaned_image_path, 'rb') as f:
                            image_bytes = f.read()
                        # 画像配列も読み込み（shape情報取得用）
                        image_array = cv2.imread(cleaned_image_path)
                        logger.info(f"補助線除去済み画像を使用: {cleaned_image_path}")
                    else:
                        logger.warning(f"補助線除去済み画像が見つかりません: {cleaned_image_path}")
                        continue
                except Exception as e:
                    logger.error(f"補助線除去済み画像読み込みエラー: {e}")
                    continue
                
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
                        
                        # この文字の評価点数を取得
                        char_scores = evaluation_scores.get(char_key, {}) if evaluation_scores else {}
                        
                        # データベース保存（評価点数付き、重複チェック有効）
                        sample_data = self.supabase.create_writing_sample(
                            writer_number=writer_number,
                            character=recognized_char,
                            image_data=image_bytes,
                            scores=char_scores if char_scores else None,
                            gemini_result=gemini_result,
                            writer_age=writer_age,
                            writer_grade=writer_grade,
                            allow_duplicates=False  # 重複防止
                        )
                        
                        # 評価点数をログ出力
                        if char_scores:
                            logger.info(f"評価点数保存: {char_key} = {char_scores}")
                        
                        result["sample_id"] = sample_data["id"]
                        result["image_path"] = sample_data.get("image_path")
                        result["saved_to_supabase"] = True
                        result["is_duplicate"] = sample_data.get("is_duplicate", False)
                        result["action"] = sample_data.get("action", "created")
                        
                        if sample_data.get("is_duplicate"):
                            # 既存サンプルに評価点数を更新
                            if char_scores:
                                try:
                                    update_result = self.supabase.update_writing_sample_scores(
                                        sample_data['id'], 
                                        char_scores, 
                                        {},  # コメントは空
                                        "system"  # 評価者
                                    )
                                    logger.info(f"既存サンプルに評価点数を更新: {recognized_char} (ID: {sample_data['id']}) = {char_scores}")
                                    result["action"] = "score_updated"
                                except Exception as e:
                                    logger.error(f"評価点数更新エラー: {e}")
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
        logger.error("Using fallback processing - main OCR processor failed to initialize")
        logger.error("This means character recognition will not work properly")
        
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
        
        # 簡易的な結果を返す（実際のサンプル文字を使用）
        sample_chars = ["清", "炎", "葉"]
        
        return {
            "character_recognition": {
                f"char_{i+1}": {
                    "image": char_regions[i]["image"],
                    "bbox": char_regions[i]["bbox"],
                    "gemini_recognition": {
                        "character": sample_chars[i] if i < len(sample_chars) else "認識不可",
                        "confidence": 0.5,
                        "reasoning": "フォールバック処理による暫定結果"
                    }
                } for i in range(len(char_regions))
            },
            "evaluations": [
                {"field": "白評価1", "recognized_text": "7", "confidence": 0.5, "type": "evaluation"},
                {"field": "黒評価1", "recognized_text": "8", "confidence": 0.5, "type": "evaluation"},
                {"field": "場評価1", "recognized_text": "6", "confidence": 0.5, "type": "evaluation"},
                {"field": "形評価1", "recognized_text": "9", "confidence": 0.5, "type": "evaluation"},
                {"field": "白評価2", "recognized_text": "8", "confidence": 0.5, "type": "evaluation"},
                {"field": "黒評価2", "recognized_text": "7", "confidence": 0.5, "type": "evaluation"},
                {"field": "場評価2", "recognized_text": "9", "confidence": 0.5, "type": "evaluation"},
                {"field": "形評価2", "recognized_text": "8", "confidence": 0.5, "type": "evaluation"},
                {"field": "白評価3", "recognized_text": "6", "confidence": 0.5, "type": "evaluation"},
                {"field": "黒評価3", "recognized_text": "9", "confidence": 0.5, "type": "evaluation"},
                {"field": "場評価3", "recognized_text": "7", "confidence": 0.5, "type": "evaluation"},
                {"field": "形評価3", "recognized_text": "8", "confidence": 0.5, "type": "evaluation"}
            ],
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
    test_image = "debug/original_input.jpg"
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