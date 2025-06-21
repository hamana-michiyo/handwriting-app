"""
改良版OCR: 画像前処理を強化
"""
import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ImprovedOCRProcessor:
    """改良版OCR処理クラス"""
    
    def __init__(self):
        self.tombo_size_range = (8, 30)
        self.tombo_area_range = (50, 900)
    
    def detect_tombo_marks(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """トンボ検出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tombo_regions = [
            (540, 190, 550, 200),   # トンボ1
            (890, 190, 900, 200),   # トンボ2  
            (540, 1400, 550, 1410), # トンボ3
            (890, 1400, 900, 1410), # トンボ4
        ]
        
        detected_tombos = []
        
        for region_idx, (x1, y1, x2, y2) in enumerate(tombo_regions):
            search_x1, search_y1 = x1 - 20, y1 - 20
            search_x2, search_y2 = x2 + 20, y2 + 20
            
            best_tombo = None
            best_score = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                if not (search_x1 <= center_x <= search_x2 and search_y1 <= center_y <= search_y2):
                    continue
                
                if not (self.tombo_size_range[0] <= w <= self.tombo_size_range[1] and 
                       self.tombo_size_range[0] <= h <= self.tombo_size_range[1]):
                    continue
                
                area = cv2.contourArea(contour)
                if not (self.tombo_area_range[0] <= area <= self.tombo_area_range[1]):
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    aspect_ratio = min(w, h) / max(w, h)
                    score = circularity * aspect_ratio
                    
                    if score > best_score:
                        best_score = score
                        best_tombo = (center_x, center_y)
            
            if best_tombo:
                detected_tombos.append(best_tombo)
                logger.info(f"トンボ{region_idx+1}検出: {best_tombo}")
        
        return detected_tombos
    
    def correct_perspective(self, image: np.ndarray, tombo_points: List[Tuple[int, int]]) -> np.ndarray:
        """透視変換"""
        if len(tombo_points) != 4:
            logger.warning("4つのトンボが必要ですが、リサイズで代替します")
            return cv2.resize(image, (800, 1200))
        
        points_sorted_by_y = sorted(tombo_points, key=lambda p: p[1])
        top_points = sorted(points_sorted_by_y[:2], key=lambda p: p[0])
        bottom_points = sorted(points_sorted_by_y[2:], key=lambda p: p[0])
        
        top_left, top_right = top_points
        bottom_left, bottom_right = bottom_points
        
        width_top = top_right[0] - top_left[0]
        width_bottom = bottom_right[0] - bottom_left[0]
        height_left = bottom_left[1] - top_left[1]
        height_right = bottom_right[1] - top_right[1]
        
        avg_width = (width_top + width_bottom) // 2
        avg_height = (height_left + height_right) // 2
        
        target_height = 1200
        target_width = int(1200 * avg_width / avg_height)
        
        src_points = np.float32([
            top_points[0], top_points[1], bottom_points[0], bottom_points[1]
        ])
        
        dst_points = np.float32([
            [0, 0], [target_width, 0], [0, target_height], [target_width, target_height]
        ])
        
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        corrected = cv2.warpPerspective(image, transform_matrix, (target_width, target_height))
        
        self.corrected_width = target_width
        self.corrected_height = target_height
        
        return corrected
    
    def extract_character_regions(self, corrected_image: np.ndarray) -> Dict:
        """補正済み画像から文字領域を抽出（画像のみ）"""
        height, width = corrected_image.shape[:2]
        
        # トンボ領域内での相対座標
        tombo_area_width = 895 - 545   # 350ピクセル
        tombo_area_height = 1405 - 195  # 1210ピクセル
        
        character_coords = [
            ("清", 600, 810, 280, 470),
            ("炎", 600, 810, 700, 900),
            ("葉", 600, 810, 1110, 1310)
        ]
        
        character_images = {}
        
        for name, x1, x2, y1, y2 in character_coords:
            # 相対座標計算
            x1_rel = (x1 - 545) / tombo_area_width
            x2_rel = (x2 - 545) / tombo_area_width
            y1_rel = (y1 - 195) / tombo_area_height
            y2_rel = (y2 - 195) / tombo_area_height
            
            # 絶対座標変換
            abs_x1 = int(x1_rel * width)
            abs_y1 = int(y1_rel * height)
            abs_x2 = int(x2_rel * width)
            abs_y2 = int(y2_rel * height)
            
            # 境界チェック
            abs_x1 = max(0, abs_x1)
            abs_y1 = max(0, abs_y1)
            abs_x2 = min(width, abs_x2)
            abs_y2 = min(height, abs_y2)
            
            if abs_x1 < abs_x2 and abs_y1 < abs_y2:
                region_image = corrected_image[abs_y1:abs_y2, abs_x1:abs_x2]
                character_images[name] = region_image
                
                # デバッグ保存
                cv2.imwrite(f"improved_char_{name}.jpg", region_image)
                logger.info(f"{name}文字領域保存: improved_char_{name}.jpg")
        
        return character_images
    
    def preprocess_number_image(self, image: np.ndarray, debug_name: str) -> List[np.ndarray]:
        """数字画像の前処理（複数バリエーション生成）"""
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        preprocessed_images = []
        
        # 1. 基本的なコントラスト強化
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 2. ガウシアンブラーでノイズ除去
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 3. 複数の二値化手法
        binary_methods = [
            ("otsu", cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("adaptive_mean", cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("adaptive_gaussian", cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("manual_light", cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)[1]),
            ("manual_dark", cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1])
        ]
        
        for method_name, binary in binary_methods:
            # 4. モルフォロジー処理でクリーンアップ
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # 5. 輪郭の穴埋め
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(cleaned)
            cv2.fillPoly(filled, contours, 255)
            
            preprocessed_images.append((f"{method_name}", filled))
            
            # デバッグ保存
            cv2.imwrite(f"improved_debug_{debug_name}_{method_name}.jpg", filled)
        
        return preprocessed_images
    
    def perform_enhanced_number_ocr(self, image: np.ndarray, region_name: str) -> Tuple[str, float]:
        """強化された数字OCR"""
        
        # 複数の前処理画像を生成
        preprocessed_images = self.preprocess_number_image(image, region_name.replace(' ', '_'))
        
        # 複数のTesseract設定
        ocr_configs = [
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789',
            r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789',  # 単一文字モード
            r'--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789',   # LSTM OCR
        ]
        
        best_result = ("", 0.0)
        all_results = []
        
        # 全ての組み合わせを試す
        for method_name, processed_image in preprocessed_images:
            for config in ocr_configs:
                try:
                    # テキスト取得
                    text = pytesseract.image_to_string(processed_image, config=config).strip()
                    
                    # 数字のみフィルタ
                    import re
                    digits_only = re.sub(r'[^0-9]', '', text)
                    
                    if digits_only:
                        # 信頼度取得
                        data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
                        confidences = [conf for conf in data['conf'] if conf > 0]
                        avg_conf = sum(confidences) / len(confidences) if confidences else 0
                        
                        result = (digits_only, avg_conf / 100.0)
                        all_results.append((method_name, config, result))
                        
                        if avg_conf > best_result[1] * 100:
                            best_result = result
                            logger.info(f"{region_name}: '{digits_only}' (信頼度: {avg_conf:.1f}) [{method_name}]")
                
                except Exception as e:
                    continue
        
        # 記入者番号の特別処理（"No. 3" -> "3"）
        if region_name == "記入者番号" and not best_result[0]:
            # "No."を除いた処理を試す
            for method_name, processed_image in preprocessed_images:
                try:
                    # より寛容な設定
                    config = r'--oem 3 --psm 6'
                    text = pytesseract.image_to_string(processed_image, config=config).strip()
                    
                    # "No. X" パターンを検索
                    import re
                    match = re.search(r'(?:No\.?\s*)?(\d+)', text, re.IGNORECASE)
                    if match:
                        digit = match.group(1)
                        best_result = (digit, 0.8)  # 高信頼度を設定
                        logger.info(f"{region_name}: '{digit}' (パターンマッチ) [{method_name}]")
                        break
                
                except Exception:
                    continue
        
        logger.info(f"{region_name} 最終結果: '{best_result[0]}' (信頼度: {best_result[1]:.2f})")
        return best_result
    
    def extract_number_regions_from_original(self, original_image: np.ndarray) -> Dict:
        """元画像から数字領域を抽出して強化OCR"""
        height, width = original_image.shape[:2]
        
        # 記入者番号と評価数字（元画像座標）
        number_regions = [
            ("記入者番号", 1800, 2300, 100, 170),
            ("白評価1", 2220, 2330, 200, 300),
            ("黒評価1", 2220, 2330, 300, 390),
            ("場評価1", 2220, 2330, 390, 470),
            ("形評価1", 2220, 2330, 460, 550),
            ("白評価2", 2220, 2330, 620, 710),
            ("黒評価2", 2220, 2330, 710, 800),
            ("場評価2", 2220, 2330, 800, 890),
            ("形評価2", 2220, 2330, 890, 990),
            ("白評価3", 2220, 2330, 1050, 1140),
            ("黒評価3", 2220, 2330, 1140, 1230),
            ("場評価3", 2220, 2330, 1220, 1310),
            ("形評価3", 2220, 2330, 1310, 1400)
        ]
        
        number_results = {
            "writer_number": {},
            "evaluations": {}
        }
        
        for name, x1, x2, y1, y2 in number_regions:
            # 境界チェック
            if x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height and x1 < x2 and y1 < y2:
                region_image = original_image[y1:y2, x1:x2]
                
                # 元画像保存
                cv2.imwrite(f"improved_num_{name.replace(' ', '_')}.jpg", region_image)
                
                # 強化OCR実行
                text, confidence = self.perform_enhanced_number_ocr(region_image, name)
                
                result_data = {
                    "text": text,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                }
                
                if name == "記入者番号":
                    number_results["writer_number"] = result_data
                else:
                    number_results["evaluations"][name] = result_data
                
            else:
                logger.warning(f"{name}: 座標が画像範囲外 ({x1},{y1},{x2},{y2})")
        
        return number_results
    
    def process_form(self, image_path: str) -> Dict:
        """改良版記入用紙処理"""
        logger.info(f"改良版処理開始: {image_path}")
        
        # 画像読み込み
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"画像が読み込めません: {image_path}")
        
        logger.info(f"元画像サイズ: {original_image.shape}")
        
        # トンボ検出
        tombo_points = self.detect_tombo_marks(original_image)
        
        # 文字用の補正済み画像
        corrected_image = self.correct_perspective(original_image, tombo_points)
        cv2.imwrite("improved_corrected.jpg", corrected_image)
        
        # 文字領域抽出（画像のみ）
        character_images = self.extract_character_regions(corrected_image)
        
        # 数字領域抽出と強化OCR（元画像から）
        number_results = self.extract_number_regions_from_original(original_image)
        
        # 結果統合
        results = {
            "tombo_detected": len(tombo_points),
            "correction_applied": len(tombo_points) == 4,
            "character_images": character_images,  # 文字画像（OCRなし）
            "writer_number": number_results["writer_number"],
            "evaluations": number_results["evaluations"]
        }
        
        logger.info("改良版処理完了")
        return results


def main():
    """改良版テスト実行"""
    
    logging.basicConfig(level=logging.INFO)
    
    processor = ImprovedOCRProcessor()
    
    try:
        results = processor.process_form("docs/記入sample.JPG")
        
        print("\n=== 改良版処理結果 ===")
        print(f"トンボ検出: {results['tombo_detected']}個")
        print(f"歪み補正: {'適用' if results['correction_applied'] else '未適用'}")
        
        print("\n[文字画像抽出]")
        for char_name in results["character_images"]:
            print(f"  {char_name}: improved_char_{char_name}.jpg 保存済み")
        
        print("\n[記入者番号]")
        writer_data = results["writer_number"]
        if writer_data:
            status = "✓" if writer_data["text"] else "✗"
            print(f"  記入者番号: '{writer_data['text']}' (信頼度: {writer_data['confidence']:.2f}) {status}")
        
        print("\n[評価数字]")
        for eval_name, eval_data in results["evaluations"].items():
            status = "✓" if eval_data["text"] else "✗"
            print(f"  {eval_name}: '{eval_data['text']}' (信頼度: {eval_data['confidence']:.2f}) {status}")
        
        print(f"\n改良版処理完了！")
        
    except Exception as e:
        logger.error(f"処理エラー: {e}")

if __name__ == "__main__":
    main()