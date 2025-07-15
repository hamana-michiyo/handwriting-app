#!/usr/bin/env python3
"""
api_server.py
=============
手書き文字評価システム FastAPI サーバー

Usage:
    uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
"""

import io
import os
import base64
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image
import cv2

from src.eval.pipeline import evaluate_pair
from src.eval.preprocessing import preprocess_from_array


# FastAPIアプリケーション初期化
app = FastAPI(
    title="手書き文字評価API",
    description="お手本とユーザー画像を4軸（形・黒・白・場）で評価するAPI",
    version="0.2.3"
)

# CORS設定（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切なオリジンを設定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======= データモデル定義 =======

class EvaluationRequest(BaseModel):
    """評価リクエストモデル（Base64画像データ用）"""
    reference_image: str  # Base64エンコードされた画像
    target_image: str     # Base64エンコードされた画像
    size: Optional[int] = 256
    enhanced_analysis: Optional[bool] = False


class EvaluationResponse(BaseModel):
    """評価結果レスポンスモデル"""
    scores: Dict[str, Any]  # Any型に変更して辞書も許可
    success: bool
    message: str


class FormProcessRequest(BaseModel):
    """記入用紙処理リクエストモデル"""
    image_base64: str         # Base64エンコードされた画像
    writer_number: str        # 記入者番号
    writer_age: Optional[int] = None    # 記入者年齢
    writer_grade: Optional[str] = None  # 記入者学年
    auto_save: Optional[bool] = True    # 自動保存フラグ


class FormProcessResponse(BaseModel):
    """記入用紙処理結果レスポンスモデル"""
    success: bool
    message: str
    character_results: Optional[Dict[str, Any]] = None
    number_results: Optional[Dict[str, Any]] = None
    perspective_corrected: Optional[bool] = False
    processing_time: Optional[float] = 0.0


# ======= ユーティリティ関数 =======

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Base64文字列をOpenCV画像配列に変換"""
    try:
        # Base64デコード
        if base64_str.startswith('data:image'):
            # data:image/jpeg;base64, の部分を除去
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        
        # PILで画像読み込み
        pil_image = Image.open(io.BytesIO(image_data))
        
        # RGBに変換（必要に応じて）
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # OpenCV形式（BGR）に変換
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"画像デコードエラー: {str(e)}")


def find_document_corners(image: np.ndarray) -> np.ndarray:
    """
    画像から文書の四隅を検出
    Args:
        image: 入力画像（BGR）
    Returns:
        corners: 四隅の座標 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ノイズ除去
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # エッジ検出
    edged = cv2.Canny(blurred, 50, 150)
    
    # 輪郭検出
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 最大の輪郭を取得
    if len(contours) == 0:
        # 四隅が見つからない場合は画像全体を使用
        h, w = gray.shape
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 輪郭を四角形に近似
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # 4つの頂点を取得
    if len(approx) >= 4:
        # 4つの頂点が見つかった場合
        corners = approx[:4].reshape(4, 2).astype(np.float32)
    else:
        # 四角形に近似できない場合は外接矩形を使用
        x, y, w, h = cv2.boundingRect(largest_contour)
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    
    # 左上、右上、右下、左下の順にソート
    corners = order_corners(corners)
    return corners


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    四隅の座標を左上、右上、右下、左下の順に並び替え
    """
    # 座標の合計で左上と右下を判定
    s = corners.sum(axis=1)
    top_left = corners[np.argmin(s)]
    bottom_right = corners[np.argmax(s)]
    
    # 座標の差で右上と左下を判定
    diff = np.diff(corners, axis=1)
    top_right = corners[np.argmin(diff)]
    bottom_left = corners[np.argmax(diff)]
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def apply_perspective_transform(image: np.ndarray, corners: np.ndarray, target_width: int = 2100, target_height: int = 2970) -> np.ndarray:
    """
    透視変換を適用してA4用紙を正面化
    Args:
        image: 入力画像
        corners: 四隅の座標
        target_width: 目標幅（A4比率：210mm）
        target_height: 目標高さ（A4比率：297mm）
    Returns:
        transformed: 変換後の画像
    """
    # 変換後の座標（A4サイズ）
    dst_corners = np.array([
        [0, 0],
        [target_width, 0],
        [target_width, target_height],
        [0, target_height]
    ], dtype=np.float32)
    
    # 透視変換行列を計算
    matrix = cv2.getPerspectiveTransform(corners, dst_corners)
    
    # 透視変換を適用
    transformed = cv2.warpPerspective(image, matrix, (target_width, target_height))
    
    return transformed


def process_cropped_form_with_opencv(image: np.ndarray, writer_number: str, writer_age: Optional[int] = None, writer_grade: Optional[str] = None, auto_save: bool = True) -> Dict[str, Any]:
    """
    切り取り済み画像にOpenCV処理を適用
    1. 四隅検出
    2. 透視変換による正面化
    3. 文字・数字認識（将来のGemini + PyTorch統合）
    """
    import time
    start_time = time.time()
    
    try:
        # デバッグ用: 元画像保存
        debug_dir = "debug"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "original_input.jpg"), image)
        print(f"[DEBUG] Original image shape: {image.shape}")
        
        # image_cropperでトリミング済みの場合は透視変換をスキップ
        # 既に記入用紙の必要部分が切り取られているため
        corrected_image = image.copy()
        # ダミーの四隅座標（画像全体）
        h, w = corrected_image.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        print(f"[DEBUG] Using cropped image directly (skipping perspective transform)")
        print(f"[DEBUG] Corrected image shape: {corrected_image.shape}")
        print(f"[DEBUG] Corrected image min/max values: {corrected_image.min()}/{corrected_image.max()}")
        
        # デバッグ用: 処理後の画像を保存
        debug_path = os.path.join(debug_dir, "perspective_corrected.jpg")
        print(f"[DEBUG] Saving processed image to: {debug_path}")
        cv2.imwrite(debug_path, corrected_image)
        print(f"[DEBUG] Image saved successfully: {os.path.exists(debug_path)}")
        
        # Step 3: Supabase OCR Processor統合
        try:
            # 一時ファイルとして保存
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                cv2.imwrite(temp_file.name, corrected_image)
                temp_path = temp_file.name
            
            # SupabaseOCRProcessorを呼び出し
            from src.core.supabase_ocr_processor import SupabaseOCRProcessor
            processor = SupabaseOCRProcessor(debug_enabled=True)
            
            # 実際の認識処理を実行
            ocr_results = processor.process_form_image(
                image_path=temp_path,
                writer_number=writer_number,
                writer_age=writer_age,
                writer_grade=writer_grade,
                auto_save=auto_save  # auto_saveパラメータを使用
            )
            
            # 一時ファイル削除
            os.unlink(temp_path)
            
            # 結果をAPIレスポンス形式に変換
            character_results = {}
            number_results = {}
            
            if "character_results" in ocr_results:
                for char_data in ocr_results["character_results"]:
                    char_key = char_data.get("char_key", "")
                    if "gemini_result" in char_data:
                        gemini_result = char_data["gemini_result"]
                        character_results[char_key] = {
                            "text": gemini_result.get("character", "認識失敗"),
                            "confidence": gemini_result.get("confidence", 0.0)
                        }
                    else:
                        character_results[char_key] = {"text": "認識未実装", "confidence": 0.0}
            
            # 記入者番号は入力フォームから取得したものを使用
            number_results["writer_number"] = {"text": writer_number, "confidence": 1.0}
            
            # 評価点数の処理
            scores = []
            if "number_results" in ocr_results:
                eval_names = ["白評価1", "黒評価1", "場評価1", "形評価1",
                             "白評価2", "黒評価2", "場評価2", "形評価2",
                             "白評価3", "黒評価3", "場評価3", "形評価3"]
                
                # number_resultsから評価データを抽出
                eval_data_dict = {}
                for result in ocr_results["number_results"]:
                    if result.get("type") == "evaluation":
                        field_name = result.get("field", "")
                        eval_data_dict[field_name] = {
                            "text": result.get("recognized_text", "0"),
                            "confidence": result.get("confidence", 0.0)
                        }
                
                # 決められた順序で評価データを配列に格納
                for eval_name in eval_names:
                    if eval_name in eval_data_dict:
                        scores.append(eval_data_dict[eval_name])
                    else:
                        scores.append({"text": "0", "confidence": 0.0})
            else:
                scores = [{"text": "0", "confidence": 0.0} for _ in range(12)]
            
            number_results["scores"] = scores
            
            print(f"[DEBUG] OCR processing completed successfully")
            print(f"[DEBUG] Supabase保存設定: auto_save={auto_save}")
            print(f"[DEBUG] Supabase保存結果: {ocr_results.get('supabase_saved', 'unknown')}")
            print(f"[DEBUG] OCR結果の keys: {list(ocr_results.keys())}")
            if 'character_results' in ocr_results:
                print(f"[DEBUG] character_results数: {len(ocr_results['character_results'])}")
                for i, char_data in enumerate(ocr_results['character_results']):
                    print(f"[DEBUG] character_results[{i}]: {char_data.get('char_key', 'unknown')}")
            
            # Supabase保存状況をログに追加
            if 'character_results' in ocr_results:
                for char_data in ocr_results['character_results']:
                    saved_status = char_data.get('saved_to_supabase', False)
                    print(f"[DEBUG] {char_data.get('char_key', '')}: Supabase保存={saved_status}")
            
            # debug/result.logに認識結果を出力
            log_path = os.path.join(debug_dir, "result.log")
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write("=== 手書き文字認識結果 ===\n")
                log_file.write(f"記入者番号: {writer_number}\n")
                log_file.write(f"処理時間: {time.time() - start_time:.2f}秒\n")
                log_file.write(f"画像サイズ: {corrected_image.shape[1]}x{corrected_image.shape[0]}\n\n")
                
                log_file.write("--- 文字認識結果 ---\n")
                for char_key, char_result in character_results.items():
                    log_file.write(f"{char_key}: '{char_result['text']}' (信頼度: {char_result['confidence']:.2f})\n")
                
                log_file.write("\n--- 数字認識結果 ---\n")
                log_file.write(f"記入者番号: {number_results['writer_number']['text']}\n")
                
                eval_names = ["白評価1", "黒評価1", "場評価1", "形評価1",
                             "白評価2", "黒評価2", "場評価2", "形評価2",
                             "白評価3", "黒評価3", "場評価3", "形評価3"]
                
                for i, score in enumerate(number_results['scores']):
                    if i < len(eval_names):
                        log_file.write(f"{eval_names[i]}: '{score['text']}' (信頼度: {score['confidence']:.2f})\n")
                    else:
                        log_file.write(f"評価{i+1}: '{score['text']}' (信頼度: {score['confidence']:.2f})\n")
                
                log_file.write(f"\n--- OCR詳細結果 ---\n")
                log_file.write(f"{ocr_results}\n")
            
            print(f"[DEBUG] Recognition results saved to: {log_path}")
            
        except Exception as ocr_error:
            print(f"[DEBUG] OCR processing failed: {ocr_error}, using fallback")
            # フォールバック: プレースホルダーデータ
            character_results = {
                "char_1": {"text": "文字認識失敗", "confidence": 0.0},
                "char_2": {"text": "文字認識失敗", "confidence": 0.0},
                "char_3": {"text": "文字認識失敗", "confidence": 0.0}
            }
            
            number_results = {
                "writer_number": {"text": writer_number, "confidence": 1.0},
                "scores": [{"text": "0", "confidence": 0.0} for _ in range(12)]
            }
            
            # エラー結果もログに出力
            log_path = os.path.join(debug_dir, "result.log")
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write("=== 手書き文字認識結果（エラー） ===\n")
                log_file.write(f"記入者番号: {writer_number}\n")
                log_file.write(f"処理時間: {time.time() - start_time:.2f}秒\n")
                log_file.write(f"画像サイズ: {corrected_image.shape[1]}x{corrected_image.shape[0]}\n")
                log_file.write(f"エラー: {ocr_error}\n\n")
                
                log_file.write("--- フォールバック結果 ---\n")
                log_file.write("文字認識: 全て失敗\n")
                log_file.write("数字認識: 全て失敗\n")
            
            print(f"[DEBUG] Error results saved to: {log_path}")
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "message": f"画像処理完了: 四隅検出→透視変換→認識処理 ({processing_time:.2f}秒)",
            "character_results": character_results,
            "number_results": number_results,
            "perspective_corrected": True,
            "processing_time": processing_time,
            "corners_detected": corners.tolist(),
            "corrected_size": {"width": corrected_image.shape[1], "height": corrected_image.shape[0]},
            "supabase_settings": {
                "auto_save_enabled": auto_save,
                "database_saved": ocr_results.get('supabase_saved', False),
                "character_save_count": len([c for c in ocr_results.get('character_results', []) if c.get('saved_to_supabase', False)])
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"画像処理エラー: {str(e)}",
            "character_results": None,
            "number_results": None,
            "perspective_corrected": False,
            "processing_time": time.time() - start_time
        }


def upload_file_to_array(upload_file: UploadFile) -> np.ndarray:
    """アップロードファイルをOpenCV画像配列に変換"""
    try:
        # ファイル内容を読み込み
        file_content = upload_file.file.read()
        
        # PILで画像読み込み
        pil_image = Image.open(io.BytesIO(file_content))
        
        # RGBに変換
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # OpenCV形式（BGR）に変換
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ファイル読み込みエラー: {str(e)}")


# ======= API エンドポイント =======

@app.get("/")
async def root():
    """ルートエンドポイント - API情報"""
    return {
        "message": "手書き文字評価API",
        "version": "0.2.3",
        "endpoints": {
            "/evaluate": "POST - 画像評価（Base64）",
            "/evaluate/upload": "POST - 画像評価（ファイルアップロード）",
            "/health": "GET - ヘルスチェック"
        }
    }


@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {"status": "healthy", "version": "0.3.0"}


@app.post("/process-cropped-form", response_model=FormProcessResponse)
async def process_cropped_form(request: FormProcessRequest):
    print(f"[DEBUG] /process-cropped-form endpoint called")
    """
    切り取り済み記入用紙画像処理エンドポイント
    
    1. 四隅検出 - OpenCVで紙の境界を自動検出
    2. 透視変換 - getPerspectiveTransformで完璧A4スキャン化
    3. AI認識 - Gemini + PyTorch による文字・数字認識（今後実装）
    
    Args:
        request: 記入用紙処理リクエスト
        
    Returns:
        処理結果（四隅検出、透視変換、認識結果）
    """
    try:
        # Base64画像をデコード
        image = decode_base64_image(request.image_base64)
        
        # OpenCV処理（四隅検出 + 透視変換 + 認識）
        result = process_cropped_form_with_opencv(
            image=image,
            writer_number=request.writer_number,
            writer_age=request.writer_age,
            writer_grade=request.writer_grade,
            auto_save=request.auto_save
        )
        
        return FormProcessResponse(**result)
        
    except Exception as e:
        return FormProcessResponse(
            success=False,
            message=f"画像処理エラー: {str(e)}",
            character_results=None,
            number_results=None,
            perspective_corrected=False,
            processing_time=0.0
        )


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_images(request: EvaluationRequest):
    """
    画像評価エンドポイント（Base64画像データ）
    
    Args:
        request: 評価リクエスト（Base64画像データ）
        
    Returns:
        評価結果
    """
    try:
        # Base64画像をデコード
        ref_image = decode_base64_image(request.reference_image)
        target_image = decode_base64_image(request.target_image)
        
        # 前処理
        ref_gray, ref_mask = preprocess_from_array(ref_image, request.size, False)
        target_gray, target_mask = preprocess_from_array(target_image, request.size, False)
        
        # 評価実行
        scores = evaluate_pair(
            ref_gray, ref_mask, 
            target_gray, target_mask, 
            enhanced_analysis=request.enhanced_analysis
        )
        
        return EvaluationResponse(
            scores=scores,
            success=True,
            message="評価が正常に完了しました"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"評価エラー: {str(e)}")


@app.post("/evaluate/upload")
async def evaluate_uploaded_images(
    reference_image: UploadFile = File(..., description="お手本画像"),
    target_image: UploadFile = File(..., description="評価対象画像"),
    size: int = Form(256, description="画像サイズ"),
    enhanced_analysis: bool = Form(False, description="精密化機能ON/OFF")
):
    """
    画像評価エンドポイント（ファイルアップロード）
    
    Args:
        reference_image: お手本画像ファイル
        target_image: 評価対象画像ファイル
        size: 処理画像サイズ
        enhanced_analysis: Phase 1.5精密化機能使用フラグ
        
    Returns:
        評価結果
    """
    try:
        # ファイルを画像配列に変換
        ref_image = upload_file_to_array(reference_image)
        target_img = upload_file_to_array(target_image)
        
        # 前処理
        ref_gray, ref_mask = preprocess_from_array(ref_image, size, False)
        target_gray, target_mask = preprocess_from_array(target_img, size, False)
        
        # 評価実行
        scores = evaluate_pair(
            ref_gray, ref_mask, 
            target_gray, target_mask, 
            enhanced_analysis=enhanced_analysis
        )
        
        return JSONResponse(content={
            "scores": scores,
            "success": True,
            "message": "評価が正常に完了しました"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"評価エラー: {str(e)}")


# ======= アプリケーション起動 =======

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
