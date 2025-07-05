"""
Gemini API用のクライアント実装
手書き文字の認識に特化
"""
import os
import base64
import logging
from typing import Optional, Dict, Any
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# .env ファイルを読み込み
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiCharacterRecognizer:
    """Gemini APIを使用した手書き文字認識クライアント"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_name = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        self.timeout = int(os.getenv('GEMINI_TIMEOUT', '30'))
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")
        
        # Gemini API の設定
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        logger.info(f"Gemini API クライアント初期化完了: {self.model_name}")
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """OpenCV画像をBase64エンコードされた文字列に変換"""
        try:
            # OpenCV (BGR) → PIL (RGB) 変換
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # PIL Image に変換
            pil_image = Image.fromarray(image_rgb)
            
            # PNG形式でバイト列に変換
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Base64エンコード
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return image_base64
            
        except Exception as e:
            logger.error(f"画像のBase64変換でエラー: {e}")
            raise
    
    def recognize_japanese_character(self, image: np.ndarray, context: str = "") -> Dict[str, Any]:
        """
        手書き日本語文字を認識
        
        Args:
            image: OpenCV形式の画像 (numpy.ndarray)
            context: 追加のコンテキスト情報
            
        Returns:
            Dict with 'character', 'confidence', 'alternatives'
        """
        try:
            # 画像をBase64に変換
            image_base64 = self.image_to_base64(image)
            
            # プロンプト構築
            prompt = self._build_character_recognition_prompt(context)
            
            # Gemini API 呼び出し
            response = self.model.generate_content([
                prompt,
                {
                    "mime_type": "image/png",
                    "data": image_base64
                }
            ])
            
            # レスポンス解析
            result = self._parse_character_response(response.text)
            
            logger.info(f"Gemini認識結果: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Gemini文字認識でエラー: {e}")
            return {
                "character": "",
                "confidence": 0.0,
                "alternatives": [],
                "error": str(e)
            }
    
    def _build_character_recognition_prompt(self, context: str) -> str:
        """文字認識用のプロンプトを構築"""
        base_prompt = """
この手書き文字の画像を解析して、書かれている日本語文字（ひらがな、カタカナ、漢字）を認識してください。

以下の形式でJSONレスポンスを返してください：
{
  "character": "認識した文字",
  "confidence": 0.95,
  "alternatives": ["代替候補1", "代替候補2", "代替候補3"],
  "reasoning": "認識の根拠や特徴"
}

注意事項：
- 1文字のみ認識してください
- 信頼度は0.0～1.0の範囲で評価
- 複数の候補がある場合は alternatives に含めてください
- 判読困難な場合は confidence を低く設定してください
- 文字が見つからない場合は character を空文字にしてください
"""
        
        if context:
            base_prompt += f"\n\n追加コンテキスト: {context}"
        
        return base_prompt
    
    def _parse_character_response(self, response_text: str) -> Dict[str, Any]:
        """Geminiのレスポンステキストを解析"""
        try:
            import json
            import re
            
            # JSONブロックを抽出
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # JSONブロックがない場合、全体からJSONを抽出を試行
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    raise ValueError("JSONレスポンスが見つかりません")
            
            # JSON解析
            result = json.loads(json_text)
            
            # 必要なフィールドの確認とデフォルト値設定
            parsed_result = {
                "character": result.get("character", ""),
                "confidence": float(result.get("confidence", 0.0)),
                "alternatives": result.get("alternatives", []),
                "reasoning": result.get("reasoning", ""),
                "raw_response": response_text
            }
            
            return parsed_result
            
        except Exception as e:
            logger.warning(f"Geminiレスポンス解析エラー: {e}")
            # フォールバック: テキストから文字を抽出
            return self._fallback_text_extraction(response_text)
    
    def _fallback_text_extraction(self, response_text: str) -> Dict[str, Any]:
        """JSON解析失敗時のフォールバック処理"""
        import re
        
        # 日本語文字を抽出
        japanese_chars = re.findall(r'[ぁ-んァ-ン一-龯]', response_text)
        
        if japanese_chars:
            character = japanese_chars[0]  # 最初の文字を採用
            confidence = 0.5  # 中程度の信頼度
        else:
            character = ""
            confidence = 0.0
        
        return {
            "character": character,
            "confidence": confidence,
            "alternatives": japanese_chars[1:4] if len(japanese_chars) > 1 else [],
            "reasoning": "フォールバック解析",
            "raw_response": response_text
        }
    
    def recognize_batch_characters(self, images: list, contexts: list = None) -> list:
        """
        複数の文字画像を一括認識
        
        Args:
            images: OpenCV画像のリスト
            contexts: 各画像に対応するコンテキストのリスト
            
        Returns:
            認識結果のリスト
        """
        if contexts is None:
            contexts = [""] * len(images)
        
        results = []
        for i, (image, context) in enumerate(zip(images, contexts)):
            logger.info(f"Gemini文字認識 {i+1}/{len(images)} 処理中...")
            result = self.recognize_japanese_character(image, context)
            results.append(result)
        
        return results
    
    def is_available(self) -> bool:
        """Gemini APIが利用可能かチェック"""
        try:
            # 簡単なテストクエリを実行
            test_response = self.model.generate_content("Hello")
            return test_response is not None
        except Exception as e:
            logger.warning(f"Gemini API利用不可: {e}")
            return False


# テスト用の関数
def test_gemini_client():
    """Geminiクライアントのテスト"""
    try:
        client = GeminiCharacterRecognizer()
        
        if not client.is_available():
            print("❌ Gemini API が利用できません")
            return False
        
        print("✅ Gemini API クライアントの初期化成功")
        print(f"使用モデル: {client.model_name}")
        print(f"タイムアウト: {client.timeout}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        return False


if __name__ == "__main__":
    test_gemini_client()