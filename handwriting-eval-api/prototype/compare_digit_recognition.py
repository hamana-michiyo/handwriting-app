import time
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import pytesseract
from pathlib import Path
import os
import sys

# digit_model_test.pyのモデルをインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from digit_model_test import SimpleCNN, predicted_image

class OCRComparator:
    def __init__(self):
        # PyTorchモデルの初期化
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load("/workspace/data/digit_model.pt"))
        self.model.eval()
        
        # Tesseractの設定
        self.tesseract_configs = [
            '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',
            '--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789',
            '--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789'
        ]
        
        # 前処理メソッド
        self.preprocessing_methods = {
            'otsu': self._otsu_threshold,
            'adaptive_mean': self._adaptive_mean,
            'adaptive_gaussian': self._adaptive_gaussian,
            'manual_light': self._manual_light,
            'manual_dark': self._manual_dark
        }

    def _otsu_threshold(self, image):
        """Otsu二値化"""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _adaptive_mean(self, image):
        """適応的平均二値化"""
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    def _adaptive_gaussian(self, image):
        """適応的ガウシアン二値化"""
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def _manual_light(self, image):
        """手動閾値（明）"""
        _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
        return binary

    def _manual_dark(self, image):
        """手動閾値（暗）"""
        _, binary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        return binary

    def pytorch_recognize(self, image_path):
        """PyTorchモデルによる数字認識"""
        start_time = time.time()
        
        # 画像読み込み
        img = Image.open(image_path).convert('L')
        
        # 予測実行
        result = predicted_image(img)
        
        end_time = time.time()
        
        return {
            'method': 'pytorch',
            'result': str(result),
            'time': end_time - start_time,
            'confidence': 1.0  # PyTorchモデルは確信度を返さないので1.0とする
        }

    def tesseract_recognize(self, image_path):
        """Tesseractによる数字認識（複数前処理・設定）"""
        start_time = time.time()
        
        # 画像読み込み
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        best_result = None
        best_confidence = 0
        
        # 複数前処理手法を試行
        for prep_name, prep_method in self.preprocessing_methods.items():
            processed_image = prep_method(image)
            
            # 複数OCR設定を試行
            for config in self.tesseract_configs:
                try:
                    # データ取得（信頼度含む）
                    data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
                    
                    # 信頼度が最も高い結果を選択
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    texts = [text.strip() for text in data['text'] if text.strip()]
                    
                    if confidences and texts:
                        max_conf_idx = confidences.index(max(confidences))
                        if max_conf_idx < len(texts):
                            text = texts[max_conf_idx]
                            confidence = confidences[max_conf_idx]
                            
                            # 数字のみを抽出
                            digit_text = ''.join(filter(str.isdigit, text))
                            
                            if digit_text and confidence > best_confidence:
                                best_result = digit_text
                                best_confidence = confidence
                                
                except Exception as e:
                    continue
        
        end_time = time.time()
        
        return {
            'method': 'tesseract',
            'result': best_result if best_result else '',
            'time': end_time - start_time,
            'confidence': best_confidence / 100.0  # 0-1の範囲に正規化
        }

    def compare_methods(self, image_directory):
        """2つの手法を比較"""
        print("数字認識手法の比較テスト")
        print("=" * 50)
        
        # 画像ファイルを取得
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(Path(image_directory).glob(ext))
        
        if not image_files:
            print(f"画像ファイルが見つかりません: {image_directory}")
            return
        
        pytorch_results = []
        tesseract_results = []
        
        for image_file in sorted(image_files):
            print(f"\n処理中: {image_file.name}")
            
            # PyTorch認識
            pytorch_result = self.pytorch_recognize(str(image_file))
            pytorch_results.append(pytorch_result)
            
            # Tesseract認識
            tesseract_result = self.tesseract_recognize(str(image_file))
            tesseract_results.append(tesseract_result)
            
            # 結果表示
            print(f"  PyTorch: {pytorch_result['result']} (時間: {pytorch_result['time']:.3f}s)")
            print(f"  Tesseract: {tesseract_result['result']} (時間: {tesseract_result['time']:.3f}s, 信頼度: {tesseract_result['confidence']:.2f})")
        
        # 統計情報
        print("\n" + "=" * 50)
        print("統計情報")
        print("=" * 50)
        
        pytorch_times = [r['time'] for r in pytorch_results]
        tesseract_times = [r['time'] for r in tesseract_results]
        
        print(f"PyTorch平均実行時間: {np.mean(pytorch_times):.3f}s")
        print(f"Tesseract平均実行時間: {np.mean(tesseract_times):.3f}s")
        
        # 結果が得られた率
        pytorch_success = sum(1 for r in pytorch_results if r['result'])
        tesseract_success = sum(1 for r in tesseract_results if r['result'])
        
        print(f"PyTorch成功率: {pytorch_success}/{len(pytorch_results)} ({pytorch_success/len(pytorch_results)*100:.1f}%)")
        print(f"Tesseract成功率: {tesseract_success}/{len(tesseract_results)} ({tesseract_success/len(tesseract_results)*100:.1f}%)")
        
        # 平均信頼度（Tesseractのみ）
        tesseract_confidences = [r['confidence'] for r in tesseract_results if r['result']]
        if tesseract_confidences:
            print(f"Tesseract平均信頼度: {np.mean(tesseract_confidences):.2f}")
        
        return pytorch_results, tesseract_results

def main():
    # 比較器を初期化
    comparator = OCRComparator()
    
    # 画像ディレクトリのパス
    image_directory = "/workspace/result/scores"
    
    # 代替パス（もしscoresディレクトリがなければ）
    if not os.path.exists(image_directory):
        image_directory = "/workspace/debug"
        if not os.path.exists(image_directory):
            print("画像ディレクトリが見つかりません")
            print("以下のディレクトリを確認してください:")
            print("- /workspace/result/scores")
            print("- /workspace/debug")
            return
    
    # 比較実行
    pytorch_results, tesseract_results = comparator.compare_methods(image_directory)

if __name__ == "__main__":
    main()