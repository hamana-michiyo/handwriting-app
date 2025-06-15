#!/usr/bin/env python3
"""
api_server.py
=============
手書き文字評価システム FastAPI サーバー

Usage:
    uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
"""

import io
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
    return {"status": "healthy", "version": "0.2.3"}


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
