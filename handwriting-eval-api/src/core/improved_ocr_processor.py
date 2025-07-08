"""
改良版OCR: page_split.pyの高精度検出ロジック統合版
"""
import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Tuple, Optional
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms

# .env ファイルを読み込み
load_dotenv()

logger = logging.getLogger(__name__)

# Gemini クライアントのインポート（オプション）
try:
    from .gemini_client import GeminiCharacterRecognizer
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        from gemini_client import GeminiCharacterRecognizer
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False
        logger.warning("Gemini API クライアントが利用できません")

logger = logging.getLogger(__name__)

class SimpleCNN(torch.nn.Module):
    """MNIST+手書き数字認識用のCNNモデル"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, 1)
        self.fc1 = torch.nn.Linear(26*26*16, 128)
        self.fc2 = torch.nn.Linear(128, 11)  # 0〜10

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 26*26*16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ImprovedOCRProcessor:
    """改良版OCR処理クラス（page_split.pyロジック統合版）"""
    
    def __init__(self, use_gemini=True, debug_dir="debug"):
        self.tombo_size_range = (8, 30)
        self.tombo_area_range = (50, 900)
        self.corrected_image = None
        self.corrected_width = None
        self.corrected_height = None
        
        # デバッグディレクトリの設定
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Gemini API クライアントの初期化
        self.gemini_client = None
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        
        if self.use_gemini:
            try:
                self.gemini_client = GeminiCharacterRecognizer()
                logger.info("Gemini API クライアント初期化成功")
            except Exception as e:
                logger.warning(f"Gemini API クライアント初期化失敗: {e}")
                self.use_gemini = False
        
        # PyTorchモデルの初期化
        self.pytorch_model = None
        self.use_pytorch = False
        try:
            self.pytorch_model = SimpleCNN()
            model_path = "/workspace/data/digit_model.pt"
            if os.path.exists(model_path):
                self.pytorch_model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.pytorch_model.eval()
                self.use_pytorch = True
                logger.info("PyTorch数字認識モデル初期化成功")
            else:
                logger.warning(f"PyTorchモデルファイルが見つかりません: {model_path}")
        except Exception as e:
            logger.warning(f"PyTorchモデル初期化失敗: {e}")
            self.use_pytorch = False
    
    def find_page_corners(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """page_split.pyのページ検出ロジック"""
        try:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 背景が濃い場合の反転
            if bw.mean() < 127:
                bw = cv2.bitwise_not(bw)
            
            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            
            page = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(page, True)
            approx = cv2.approxPolyDP(page, 0.02 * peri, True)
            
            if len(approx) != 4:
                return None
            
            return approx.reshape(4, 2).astype("float32")
        except Exception as e:
            logger.warning(f"ページ検出失敗: {e}")
            return None
    
    def order_corners(self, pts: np.ndarray) -> np.ndarray:
        """4点をTL, TR, BR, BL順に並べる"""
        pts = pts[np.argsort(pts[:, 0])]  # x で左2 / 右2
        left, right = pts[:2], pts[2:]
        tl = left[np.argmin(left[:, 1])]
        bl = left[np.argmax(left[:, 1])]
        tr = right[np.argmin(right[:, 1])]
        br = right[np.argmax(right[:, 1])]
        return np.array([tl, tr, br, bl], dtype="float32")
    
    def perspective_correct_advanced(self, img: np.ndarray, corners: np.ndarray, dbg=False) -> np.ndarray:
        """page_split.pyの透視変換ロジック"""
        pts = self.order_corners(corners)
        
        wA = np.linalg.norm(pts[2] - pts[3])
        wB = np.linalg.norm(pts[1] - pts[0])
        hA = np.linalg.norm(pts[1] - pts[2])
        hB = np.linalg.norm(pts[0] - pts[3])
        W = int(max(wA, wB))
        H = int(max(hA, hB))
        
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (W, H))
        
        if dbg:
            dbg_img = img.copy()
            for (x, y) in pts.astype(int):
                cv2.circle(dbg_img, (x, y), 10, (0, 0, 255), -1)
            cv2.imwrite(str(self.debug_dir / "dbg_corners.jpg"), dbg_img)
            cv2.imwrite(str(self.debug_dir / "dbg_warped.jpg"), warped)
        
        return warped
    
    def correct_perspective(self, image: np.ndarray, debug=False) -> np.ndarray:
        """page_split.pyベースの透視変換"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # ページ検出を試行
        corners = self.find_page_corners(gray)
        
        if corners is not None:
            # 透視変換実行
            warped = self.perspective_correct_advanced(image, corners, debug)
            logger.info(f"透視変換適用: {warped.shape}")
            return warped
        else:
            # 補正スキップ
            logger.info("ページ検出失敗、補正スキップ")
            return image.copy()
    
    def detect_char_cells(self, gray: np.ndarray, debug=False) -> List[Tuple[int, int, int, int]]:
        """page_split.pyの文字セル検出ロジック"""
        try:
            H, W = gray.shape
            
            # 二値化 → 細線を太らせて輪郭を閉じる
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)
            kernel = np.ones((3, 3), np.uint8)
            th = cv2.dilate(th, kernel, iterations=1)
            
            # 輪郭抽出
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # ほぼ正方形・面積闾値でフィルタ
            cand = []
            for c in cnts:
                area = cv2.contourArea(c)
                if area < (H * W) * 0.002 or area > (H * W) * 0.03:
                    continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) != 4:
                    continue
                x, y, w, h = cv2.boundingRect(approx)
                if 0.85 < w / h < 1.15:  # 正方形っぽい
                    cand.append((x, y, w, h))
            
            if len(cand) < 3:
                raise RuntimeError("課題マスらしき四角が 3 つ見つかりません")
            
            # 列ごとにグルーピング → 右列を選択
            xs = np.array([[c[0]] for c in cand], np.float32)
            _, labels, centers = cv2.kmeans(xs, 2, None,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                10, cv2.KMEANS_PP_CENTERS)
            right_cluster = np.argmax(centers)  # x が大きい方
            right_boxes = [c for c,l in zip(cand, labels.flatten()) if l == right_cluster]
            
            # y でソートして上→下 3 つを取得
            right_boxes = sorted(right_boxes, key=lambda b: b[1])[:3]
            cells = []
            margin = 10
            for x, y, w, h in right_boxes:
                cells.append((x + margin, y + margin, x + w - margin, y + h - margin))
            
            if debug:
                dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                for (x1,y1,x2,y2) in cells:
                    cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.imwrite(str(self.debug_dir / "dbg_cells_contour.jpg"), dbg)
            
            if len(cells) != 3:
                raise RuntimeError("課題マスを 3 つ取得できません")
            
            return cells
        
        except Exception as e:
            logger.warning(f"文字セル検出失敗: {e}")
            # フォールバック: 固定比率
            H, W = gray.shape
            W0 = int(W*0.07); W1 = int(W*0.27)
            cell_h = int(H*0.23); gap = int(H*0.31)
            cells = [(W0, int(H*0.11)+i*gap, W1, int(H*0.11)+i*gap+cell_h) for i in range(3)]
            return cells
    
    def recognize_character_with_gemini(self, image: np.ndarray, char_name: str) -> Dict[str, any]:
        """Gemini APIを使用した文字認識"""
        if not self.use_gemini or self.gemini_client is None:
            return {
                "character": "",
                "confidence": 0.0,
                "method": "gemini_unavailable"
            }
        
        try:
            context = f"手書き練習用紙の文字枠{char_name}から切り出された手書き文字"            
            result = self.gemini_client.recognize_japanese_character(image, context)
            
            return {
                "character": result.get("character", ""),
                "confidence": result.get("confidence", 0.0),
                "alternatives": result.get("alternatives", []),
                "reasoning": result.get("reasoning", ""),
                "method": "gemini"
            }
            
        except Exception as e:
            logger.error(f"Gemini文字認識エラー ({char_name}): {e}")
            return {
                "character": "",
                "confidence": 0.0,
                "method": "gemini_error",
                "error": str(e)
            }
    
    def extract_character_regions(self, corrected_image: np.ndarray, debug=False) -> Dict:
        """page_split.pyベースの文字領域抽出 + Gemini文字認識"""
        gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY) if len(corrected_image.shape) == 3 else corrected_image
        
        # 文字セル検出
        cells = self.detect_char_cells(gray, debug)
        
        character_results = {}
        char_names = ["char_1", "char_2", "char_3"]  # 汎用名
        
        for i, (x1, y1, x2, y2) in enumerate(cells):
            name = char_names[i]
            region_image = gray[y1:y2, x1:x2]
            
            # 補助線除去を適用
            cleaned_region = self.remove_guidelines(region_image, save_debug=True, debug_name=name)
            
            # デバッグ保存（補助線除去前）
            char_file = self.debug_dir / f"improved_char_{name}_original.jpg"
            cv2.imwrite(str(char_file), region_image)
            
            # デバッグ保存（補助線除去後）
            char_file_cleaned = self.debug_dir / f"improved_char_{name}.jpg"
            cv2.imwrite(str(char_file_cleaned), cleaned_region)
            logger.info(f"{name}文字領域保存（補助線除去済み）: {char_file_cleaned}")
            
            # Gemini文字認識（補助線除去後の画像を使用）
            gemini_result = self.recognize_character_with_gemini(cleaned_region, name)
            
            character_results[name] = {
                "image": region_image,  # 元画像
                "cleaned_image": cleaned_region,  # 補助線除去後
                "bbox": (x1, y1, x2, y2),
                "gemini_recognition": gemini_result
            }
            
            # 認識結果をログ出力
            if gemini_result["character"]:
                confidence = gemini_result["confidence"]
                char = gemini_result["character"]
                logger.info(f"{name} Gemini認識: '{char}' (信頼度: {confidence:.2f})")
            else:
                logger.info(f"{name} Gemini認識: 認識失敗")
        
        return character_results
    
    def remove_guidelines(self, image: np.ndarray, save_debug=False, debug_name="") -> np.ndarray:
        """補助線除去（元手法ベース）"""
        try:
            # グレースケール変換
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # ガウシアンブラー
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # 二値化（固定閾値127）
            _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
            
            # モルフォロジー（軽め）
            kernel = np.ones((2, 2), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 反転して背景白・文字黒
            result = cv2.bitwise_not(opening)
            
            # デバッグ保存
            if save_debug and debug_name:
                debug_file = self.debug_dir / f"guideline_removed_{debug_name}.jpg"
                cv2.imwrite(str(debug_file), result)
                logger.info(f"補助線除去結果保存: {debug_file}")
            
            return result
            
        except Exception as e:
            logger.warning(f"補助線除去エラー: {e}")
            # エラー時は元画像をそのまま返す
            return image
    
    def pytorch_digit_recognition(self, image: np.ndarray, region_name: str) -> Tuple[str, float]:
        """PyTorchモデルによる数字認識"""
        if not self.use_pytorch:
            return "", 0.0
        
        try:
            # OpenCV画像をPILに変換
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image).convert('L')
            else:
                pil_image = Image.fromarray(image).convert('L')
            
            # 背景白・文字黒なら反転（MNISTに合わせる）
            pil_image = ImageOps.invert(pil_image)
            
            # 余白を自動クロップ
            bbox = pil_image.getbbox()
            if bbox:
                pil_image = pil_image.crop(bbox)
            
            # 28x28にリサイズ
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            input_tensor = transform(pil_image).unsqueeze(0)
            
            # 推論
            with torch.no_grad():
                output = self.pytorch_model(input_tensor)
                _, predicted = torch.max(output.data, 1)
                result = predicted.item()
                
                # 確信度を計算（softmax）
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence = probabilities[0][result].item()
                
                logger.info(f"{region_name}: PyTorch認識 '{result}' (信頼度: {confidence:.2f})")
                return str(result), confidence
                
        except Exception as e:
            logger.error(f"PyTorch認識エラー ({region_name}): {e}")
            return "", 0.0
    
    def preprocess_number_image(self, image: np.ndarray, debug_name: str) -> List[np.ndarray]:
        """数字画像の前処理（複数バリエーション生成）"""
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 画像サイズチェックとリサイズ
        h, w = gray.shape
        if h < 30 or w < 30:
            # 小さすぎる場合は拡大
            scale_factor = max(30 / h, 30 / w, 3.0)  # 最低3倍拡大
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.info(f"{debug_name}: 画像を{scale_factor:.1f}倍拡大 {w}x{h} -> {new_w}x{new_h}")
        
        preprocessed_images = []
        
        # 元画像も追加（前処理なし）
        preprocessed_images.append(("original", gray))
        
        # 1. 基本的なコントラスト強化
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 2. ガウシアンブラーでノイズ除去
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 3. 複数の二値化手法
        binary_methods = [
            ("otsu", cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            #("adaptive_mean", cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)),
            #("adaptive_gaussian", cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("manual_light", cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]),
            ("manual_dark", cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)[1])
        ]
        
        for method_name, binary in binary_methods:
            # 空の画像チェック
            if binary is None or binary.size == 0:
                continue
            
            # 4. 軽いモルフォロジー処理
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # ピクセル数チェック
            white_pixels = cv2.countNonZero(cleaned)
            if white_pixels < 10:  # 白ピクセルが少なすぎる
                continue
            
            preprocessed_images.append((f"{method_name}", cleaned))
            
            # デバッグ保存
            debug_file = self.debug_dir / f"improved_debug_{debug_name}_{method_name}.jpg"
            cv2.imwrite(str(debug_file), cleaned)
        
        return preprocessed_images
    
    def perform_enhanced_number_ocr(self, image: np.ndarray, region_name: str) -> Tuple[str, float]:
        """強化された数字OCR（PyTorchモデル優先）"""
        
        # PyTorchモデルを最初に試す
        if self.use_pytorch:
            pytorch_result, pytorch_conf = self.pytorch_digit_recognition(image, region_name)
            if pytorch_result and pytorch_conf > 0.3:  # 信頼度閾値
                logger.info(f"{region_name}: PyTorch成功 '{pytorch_result}' (信頼度: {pytorch_conf:.2f})")
                return pytorch_result, pytorch_conf
            else:
                logger.info(f"{region_name}: PyTorch失敗、Tesseractにフォールバック")
        
        # Tesseractフォールバック
        # 複数の前処理画像を生成
        preprocessed_images = self.preprocess_number_image(image, region_name.replace(' ', '_'))
        
        # 複数のTesseract設定（緩い設定を追加）
        ocr_configs = [
            r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789',  # 単一文字モード
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',   # 単語モード
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',   # 単一テキスト行
            r'--oem 3 --psm 6',  # ホワイトリストなし
            r'--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789',  # LSTM + 単一文字
            r'--oem 1 --psm 8',  # LSTM + ホワイトリストなし
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
                    
                    # 生テキストも記録（デバッグ用）
                    if text and len(text.strip()) > 0:
                        logger.debug(f"{region_name} [{method_name}]: 生テキスト='{text.strip()}'")
                    
                    digits_only = re.sub(r'[^0-9]', '', text)
                    
                    if digits_only:
                        # 信頼度取得
                        try:
                            data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
                            confidences = [conf for conf in data['conf'] if conf > 0]
                            avg_conf = sum(confidences) / len(confidences) if confidences else 0
                        except:
                            avg_conf = 50  # フォールバック信頼度
                        
                        result = (digits_only, avg_conf / 100.0)
                        all_results.append((method_name, config, result))
                        
                        if avg_conf > best_result[1] * 100:
                            best_result = result
                            logger.info(f"{region_name}: '{digits_only}' (信頼度: {avg_conf:.1f}) [{method_name}]")
                    
                    # 数字がなくてもテキストがある場合は記録
                    elif text and len(text.strip()) > 0:
                        logger.debug(f"{region_name} [{method_name}]: 数字以外='{text.strip()}'")
                
                except Exception as e:
                    logger.debug(f"{region_name} [{method_name}] OCRエラー: {e}")
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
        
        logger.info(f"{region_name} 最終結果: '{best_result[0]}' (信頼度: {best_result[1]:.2f}) [Tesseract]")
        return best_result
    
    def detect_score_and_comment_boxes(self, gray: np.ndarray, debug=False) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """page_split.pyの点数・コメント枠検出ロジック"""
        H, W = gray.shape
        roi_x0 = int(W * 0.6)  # 右 40% だけ見る
        roi = gray[:, roi_x0:]
        
        # 二値化 & 前処理
        th = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 6)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)
        th = cv2.dilate(th, np.ones((3, 3), np.uint8), 1)
        
        # 輪郭
        cnts, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        score_cand, cmt_cand = [], []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < 20 or h < 30:  # 小さすぎるもの除外
                continue
            
            ratio = w / h
            area = w * h
            if area < 600:  # 30×30
                continue
            
            # (x,y) をフル画像座標系に直す
            X = x + roi_x0
            
            # 点数候補: 正方形〜やや縦長
            if 0.8 < ratio < 1.3:
                score_cand.append((X, y, w, h))
            # コメント候補: 横長
            elif ratio > 4.5:
                cmt_cand.append((X, y, w, h))
        
        if not score_cand or not cmt_cand:
            raise RuntimeError("点数枠 or コメント枠が検出できません")
        
        if debug:
            dbg_roi = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
            for X, Y, w, h in score_cand:
                cv2.rectangle(dbg_roi, (X - roi_x0, Y), (X - roi_x0 + w, Y + h), (0, 255, 255), 1)
            cv2.imwrite(str(self.debug_dir / "dbg_score_candidates.jpg"), dbg_roi)
        
        # 右端列だけ残す
        def keep_rightmost(boxes, tol=25):
            max_x = max(b[0] for b in boxes)
            return [b for b in boxes if abs(b[0] - max_x) < tol]
        
        score_cand = keep_rightmost(score_cand, 40)  # 40px 以内
        cmt_cand = keep_rightmost(cmt_cand)
        
        # y 昇順に 12 個ずつそろえる
        score_cand = sorted(score_cand, key=lambda b: b[1])[:12]
        cmt_cand = sorted(cmt_cand, key=lambda b: b[1])[:12]
        
        # margin を内側へ入れて (x1,y1,x2,y2) に変換
        def to_box(lst, margin=4):
            return [(x+margin, y+margin, x+w-margin, y+h-margin) for (x,y,w,h) in lst]
        
        score_boxes = to_box(score_cand)
        cmt_boxes = to_box(cmt_cand)
        
        # デバッグ描画
        if debug:
            dbg_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for x1,y1,x2,y2 in score_boxes:
                cv2.rectangle(dbg_img, (x1,y1), (x2,y2), (0,0,255), 2)  # 赤=点数
            for x1,y1,x2,y2 in cmt_boxes:
                cv2.rectangle(dbg_img, (x1,y1), (x2,y2), (255,0,0), 2)  # 青=コメント
            cv2.imwrite(str(self.debug_dir / "dbg_score_comment_boxes.jpg"), dbg_img)
        
        if len(score_boxes) != 12 or len(cmt_boxes) != 12:
            raise RuntimeError(f"検出数  score:{len(score_boxes)}  cmt:{len(cmt_boxes)}")
        
        return score_boxes, cmt_boxes
    
    def extract_writer_id_region(self, corrected_image: np.ndarray) -> np.ndarray:
        """page_split.pyベースの記入者番号抽出"""
        height, width = corrected_image.shape[:2]
        # 固定比率で記入者番号領域を抽出
        x, y, w, h = (0.73, 0.03, 0.22, 0.05)
        x1, y1 = int(width * x), int(height * y)
        x2, y2 = int(width * (x + w)), int(height * (y + h))
        return corrected_image[y1:y2, x1:x2]
    
    def extract_number_regions_from_original(self, original_image: np.ndarray, debug=False) -> Dict:
        """動的点数・コメント枠抽出 + 記入者番号抽出"""
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) if len(original_image.shape) == 3 else original_image
        
        # 透視変換適用
        corrected = self.correct_perspective(original_image, debug)
        corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if len(corrected.shape) == 3 else corrected
        
        number_results = {
            "writer_number": {},
            "evaluations": {},
            "comments": {}
        }
        
        try:
            # 記入者番号抽出
            writer_region = self.extract_writer_id_region(corrected_gray)
            cv2.imwrite(str(self.debug_dir / "improved_writer_id.jpg"), writer_region)
            text, confidence = self.perform_enhanced_number_ocr(writer_region, "記入者番号")
            number_results["writer_number"] = {
                "text": text,
                "confidence": confidence,
                "bbox": "dynamic"
            }
            
            # 点数・コメント枠検出
            score_boxes, cmt_boxes = self.detect_score_and_comment_boxes(corrected_gray, debug)
            
            # 点数枠処理
            eval_names = ["白評価1", "黒評価1", "場評価1", "形評価1",
                         "白評価2", "黒評価2", "場評価2", "形評価2",
                         "白評価3", "黒評価3", "場評価3", "形評価3"]
            
            for idx, (x1, y1, x2, y2) in enumerate(score_boxes):
                if idx < len(eval_names):
                    name = eval_names[idx]
                    region_image = corrected_gray[y1:y2, x1:x2]
                    score_file = self.debug_dir / f"improved_score_{idx+1}.jpg"
                    cv2.imwrite(str(score_file), region_image)
                    text, confidence = self.perform_enhanced_number_ocr(region_image, name)
                    number_results["evaluations"][name] = {
                        "text": text,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2)
                    }
            
            # コメント枠処理
            comment_names = ["白コメント1", "黒コメント1", "場コメント1", "形コメント1",
                           "白コメント2", "黒コメント2", "場コメント2", "形コメント2",
                           "白コメント3", "黒コメント3", "場コメント3", "形コメント3"]
            
            for idx, (x1, y1, x2, y2) in enumerate(cmt_boxes):
                if idx < len(comment_names):
                    name = comment_names[idx]
                    region_image = corrected_gray[y1:y2, x1:x2]
                    comment_file = self.debug_dir / f"improved_comment_{idx+1}.jpg"
                    cv2.imwrite(str(comment_file), region_image)
                    # コメントはOCRしないで保存のみ
                    number_results["comments"][name] = {
                        "bbox": (x1, y1, x2, y2),
                        "image_saved": str(comment_file)
                    }
        
        except Exception as e:
            logger.warning(f"動的検出失敗: {e}")
            # フォールバック: 空の結果を返す
        
        return number_results
    
    def process_form(self, image_path: str, debug=False) -> Dict:
        """改良版記入用紙処理（page_split.pyロジック統合版）"""
        logger.info(f"改良版処理開始: {image_path}")
        
        # 画像読み込み
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"画像が読み込めません: {image_path}")
        
        logger.info(f"元画像サイズ: {original_image.shape}")
        
        # 透視変換適用
        corrected_image = self.correct_perspective(original_image, debug)
        cv2.imwrite(str(self.debug_dir / "improved_corrected.jpg"), corrected_image)
        
        # 文字領域抽出（動的検出）
        character_images = self.extract_character_regions(corrected_image, debug)
        
        # 数字・コメント領域抽出とOCR（動的検出）
        number_results = self.extract_number_regions_from_original(original_image, debug)
        
        # 結果統合
        results = {
            "correction_applied": True,
            "character_recognition": character_images,  # Gemini認識結果含む
            "writer_number": number_results["writer_number"],
            "evaluations": number_results["evaluations"],
            "comments": number_results["comments"],
            "gemini_enabled": self.use_gemini,
            "pytorch_enabled": self.use_pytorch
        }
        
        logger.info("改良版処理完了")
        return results


def main():
    """改良版テスト実行（page_split.py統合 + Gemini認識）"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Gemini利用可否をチェック
    use_gemini = os.getenv('GEMINI_API_KEY') is not None
    if use_gemini:
        print("🚀 Gemini API有効化: 文字認識にGeminiを使用")
    else:
        print("⚠️ Gemini API無効: .envファイルにGEMINI_API_KEYを設定してください")
    
    # PyTorchモデル利用可否をチェック
    pytorch_model_path = "/workspace/data/digit_model.pt"
    if os.path.exists(pytorch_model_path):
        print("🧠 PyTorch数字認識モデル有効化: 高精度数字認識を使用")
    else:
        print("⚠️ PyTorchモデル無効: digit_model.ptが見つかりません")
    
    processor = ImprovedOCRProcessor(use_gemini=use_gemini)
    
    try:
        results = processor.process_form("docs/記入sample.JPG", debug=True)
        
        print("\n=== 改良版処理結果（page_split.py統合 + Gemini認識） ===")
        print(f"歪み補正: {'適用' if results['correction_applied'] else '未適用'}")
        
        print("\n[文字画像抽出・認識（動的検出 + Gemini）]")
        for char_name, char_data in results["character_recognition"].items():
            print(f"  {char_name}: debug/improved_char_{char_name}.jpg 保存済み")
            
            # Gemini認識結果を表示
            gemini_result = char_data.get("gemini_recognition", {})
            if gemini_result.get("character"):
                char = gemini_result["character"]
                conf = gemini_result["confidence"]
                status = "✓" if conf > 0.5 else "⚠"
                print(f"    → Gemini認識: '{char}' (信頼度: {conf:.2f}) {status}")
                
                # 代替候補があれば表示
                alternatives = gemini_result.get("alternatives", [])
                if alternatives:
                    print(f"    → 代替候補: {', '.join(alternatives[:3])}")
            else:
                method = gemini_result.get("method", "unknown")
                print(f"    → Gemini認識: 失敗 ({method})")
        
        print("\n[記入者番号（動的検出）]")
        writer_data = results["writer_number"]
        if writer_data:
            status = "✓" if writer_data["text"] else "✗"
            print(f"  記入者番号: '{writer_data['text']}' (信頼度: {writer_data['confidence']:.2f}) {status}")
        
        print("\n[評価数字（動的検出）]")
        for eval_name, eval_data in results["evaluations"].items():
            status = "✓" if eval_data["text"] else "✗"
            print(f"  {eval_name}: '{eval_data['text']}' (信頼度: {eval_data['confidence']:.2f}) {status}")
        
        print("\n[コメント枠（動的検出）]")
        for comment_name, comment_data in results["comments"].items():
            saved_path = Path(comment_data['image_saved'])
            print(f"  {comment_name}: {saved_path} 保存済み")
        
        print(f"\n改良版処理完了！（動的検出 + Gemini文字認識 + PyTorch数字認識統合）")
        print(f"Gemini API使用: {'✅' if results.get('gemini_enabled') else '❌'}")
        print(f"PyTorch数字認識使用: {'✅' if results.get('pytorch_enabled') else '❌'}")
        
    except Exception as e:
        logger.error(f"処理エラー: {e}")

if __name__ == "__main__":
    main()