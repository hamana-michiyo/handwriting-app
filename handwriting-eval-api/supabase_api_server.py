"""
Supabase統合 FastAPI サーバー
- SupabaseOCRProcessor のAPI化
- Flutter アプリとの連携
- RESTful エンドポイント提供
Created: 2025-01-06
"""

import os
import tempfile
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import base64
import io
from PIL import Image
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# プロジェクト内モジュール
from src.core.supabase_ocr_processor import SupabaseOCRProcessor
from src.database.supabase_client import SupabaseClient

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# Pydantic モデル定義
# ===========================

class ProcessFormRequest(BaseModel):
    """記入用紙処理リクエスト"""
    image_base64: str = Field(..., description="Base64エンコードされた画像データ")
    writer_number: str = Field(..., description="記入者番号", example="writer_001")
    writer_age: Optional[int] = Field(None, description="記入者年齢", example=20)
    writer_grade: Optional[str] = Field(None, description="記入者学年", example="大学")
    auto_save: bool = Field(True, description="自動保存フラグ")

class ProcessFormResponse(BaseModel):
    """記入用紙処理レスポンス"""
    success: bool = Field(..., description="処理成功フラグ")
    message: str = Field(..., description="処理メッセージ")
    character_results: List[Dict[str, Any]] = Field(default=[], description="文字認識結果")
    number_results: List[Dict[str, Any]] = Field(default=[], description="数字認識結果")
    writer_number: str = Field(..., description="記入者番号")
    processing_timestamp: str = Field(..., description="処理タイムスタンプ")
    database_stats: Optional[Dict[str, Any]] = Field(None, description="データベース統計")

class UpdateScoresRequest(BaseModel):
    """評価スコア更新リクエスト"""
    scores: Dict[str, int] = Field(..., description="評価スコア", example={"white": 8, "black": 7, "center": 9, "shape": 8})
    comments: Optional[Dict[str, str]] = Field(None, description="コメント", example={"white": "Good balance", "black": "Nice thickness"})
    evaluator: Optional[str] = Field(None, description="評価者", example="teacher_001")

class UpdateScoresResponse(BaseModel):
    """評価スコア更新レスポンス"""
    success: bool = Field(..., description="更新成功フラグ")
    message: str = Field(..., description="更新メッセージ")
    sample_id: int = Field(..., description="更新されたサンプルID")
    updated_data: Dict[str, Any] = Field(..., description="更新されたデータ")

class WriterSamplesResponse(BaseModel):
    """記入者サンプル一覧レスポンス"""
    success: bool = Field(..., description="取得成功フラグ")
    writer_number: str = Field(..., description="記入者番号")
    samples: List[Dict[str, Any]] = Field(..., description="サンプルリスト")
    count: int = Field(..., description="サンプル数")

class StatsResponse(BaseModel):
    """統計情報レスポンス"""
    success: bool = Field(..., description="取得成功フラグ")
    stats: Dict[str, Any] = Field(..., description="統計情報")
    timestamp: str = Field(..., description="取得タイムスタンプ")

# ===========================
# FastAPI アプリケーション初期化
# ===========================

app = FastAPI(
    title="手書き文字評価 API",
    description="Gemini AI + Supabase統合による手書き文字認識・評価システム",
    version="0.6.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS設定（Flutter等のフロントエンドからのアクセス許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # プロダクションでは適切なオリジンを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数
supabase_processor: Optional[SupabaseOCRProcessor] = None

# ===========================
# 初期化・ヘルスチェック
# ===========================

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    global supabase_processor
    
    try:
        # Supabase OCRプロセッサ初期化
        supabase_processor = SupabaseOCRProcessor(
            supabase_url=os.getenv('SUPABASE_URL'),
            supabase_key=os.getenv('SUPABASE_KEY'),
            bucket_name=os.getenv('SUPABASE_BUCKET', 'ml-data'),
            debug_enabled=os.getenv('DEBUG_ENABLED', 'false').lower() == 'true'
        )
        
        logger.info("Supabase OCR Processor initialized successfully")
        
        # データベース接続テスト
        stats = supabase_processor.get_database_stats()
        logger.info(f"Database connection verified. Stats: {stats}")
        
    except Exception as e:
        logger.error(f"Failed to initialize Supabase OCR Processor: {e}")
        raise

@app.get("/health", summary="ヘルスチェック")
async def health_check():
    """サーバーヘルスチェック"""
    try:
        if supabase_processor is None:
            raise HTTPException(status_code=503, detail="Supabase processor not initialized")
        
        # データベース接続テスト
        stats = supabase_processor.get_database_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database_status": "connected",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.get("/", summary="API情報")
async def root():
    """API基本情報"""
    return {
        "name": "手書き文字評価 API",
        "version": "0.6.0",
        "description": "Gemini AI + Supabase統合による手書き文字認識・評価システム",
        "endpoints": {
            "POST /process-form": "記入用紙画像処理",
            "POST /process-form/upload": "記入用紙ファイルアップロード処理", 
            "GET /samples/{writer_number}": "記入者別サンプル取得",
            "PUT /samples/{sample_id}/scores": "評価スコア更新",
            "GET /stats": "統計情報取得",
            "GET /health": "ヘルスチェック",
            "GET /docs": "Swagger UIドキュメント"
        },
        "features": [
            "Gemini AI文字認識 (99%精度)",
            "Tesseract数字認識",
            "Supabase データベース統合",
            "画像ストレージ管理",
            "重複防止機能",
            "リアルタイム統計"
        ]
    }

# ===========================
# メイン処理エンドポイント
# ===========================

@app.post("/process-form", response_model=ProcessFormResponse, summary="記入用紙処理（Base64）")
async def process_form_base64(request: ProcessFormRequest):
    """
    Base64エンコードされた記入用紙画像を処理
    
    - **image_base64**: Base64エンコードされた画像データ
    - **writer_number**: 記入者番号（例: "writer_001"）
    - **writer_age**: 記入者年齢（オプション）
    - **writer_grade**: 記入者学年（オプション）
    - **auto_save**: 自動保存フラグ（デフォルト: true）
    
    Returns:
        処理結果（文字認識結果、数字認識結果、統計情報）
    """
    if supabase_processor is None:
        raise HTTPException(status_code=503, detail="Supabase processor not initialized")
    
    try:
        # Base64画像デコード
        try:
            # data:image/jpeg;base64,プレフィックスを除去
            if "," in request.image_base64:
                image_data = base64.b64decode(request.image_base64.split(",")[1])
            else:
                image_data = base64.b64decode(request.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
        
        # 一時ファイル作成
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name
        
        try:
            # OCR処理実行
            results = supabase_processor.process_form_image(
                image_path=temp_path,
                writer_number=request.writer_number,
                writer_age=request.writer_age,
                writer_grade=request.writer_grade,
                auto_save=request.auto_save
            )
            
            # エラーチェック
            if "error" in results:
                raise HTTPException(status_code=500, detail=f"Processing failed: {results['error']}")
            
            # レスポンス構築
            response = ProcessFormResponse(
                success=True,
                message=f"Form processed successfully. {len(results.get('character_results', []))} characters recognized.",
                character_results=results.get('character_results', []),
                number_results=results.get('number_results', []),
                writer_number=request.writer_number,
                processing_timestamp=results.get('processing_timestamp', datetime.now().isoformat()),
                database_stats=results.get('database_stats')
            )
            
            logger.info(f"Form processing completed for writer: {request.writer_number}")
            return response
            
        finally:
            # 一時ファイル削除
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing form: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/process-form/upload", response_model=ProcessFormResponse, summary="記入用紙処理（ファイルアップロード）")
async def process_form_upload(
    file: UploadFile = File(..., description="記入用紙画像ファイル"),
    writer_number: str = Form(..., description="記入者番号"),
    writer_age: Optional[int] = Form(None, description="記入者年齢"),
    writer_grade: Optional[str] = Form(None, description="記入者学年"),
    auto_save: bool = Form(True, description="自動保存フラグ")
):
    """
    ファイルアップロードによる記入用紙画像処理
    
    - **file**: 記入用紙画像ファイル（JPEG/PNG）
    - **writer_number**: 記入者番号
    - **writer_age**: 記入者年齢（オプション）
    - **writer_grade**: 記入者学年（オプション）
    - **auto_save**: 自動保存フラグ
    
    Returns:
        処理結果（文字認識結果、数字認識結果、統計情報）
    """
    if supabase_processor is None:
        raise HTTPException(status_code=503, detail="Supabase processor not initialized")
    
    # ファイル形式チェック
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # 一時ファイル作成
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # OCR処理実行
            results = supabase_processor.process_form_image(
                image_path=temp_path,
                writer_number=writer_number,
                writer_age=writer_age,
                writer_grade=writer_grade,
                auto_save=auto_save
            )
            
            # エラーチェック
            if "error" in results:
                raise HTTPException(status_code=500, detail=f"Processing failed: {results['error']}")
            
            # レスポンス構築
            response = ProcessFormResponse(
                success=True,
                message=f"Form processed successfully from file: {file.filename}. {len(results.get('character_results', []))} characters recognized.",
                character_results=results.get('character_results', []),
                number_results=results.get('number_results', []),
                writer_number=writer_number,
                processing_timestamp=results.get('processing_timestamp', datetime.now().isoformat()),
                database_stats=results.get('database_stats')
            )
            
            logger.info(f"File upload processing completed for writer: {writer_number}, file: {file.filename}")
            return response
            
        finally:
            # 一時ファイル削除
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===========================
# データ取得・管理エンドポイント
# ===========================

@app.get("/samples/{writer_number}", response_model=WriterSamplesResponse, summary="記入者別サンプル取得")
async def get_writer_samples(writer_number: str):
    """
    指定した記入者のサンプル一覧を取得
    
    Args:
        writer_number: 記入者番号
        
    Returns:
        記入者のサンプルリスト
    """
    if supabase_processor is None:
        raise HTTPException(status_code=503, detail="Supabase processor not initialized")
    
    try:
        samples = supabase_processor.get_writer_samples(writer_number)
        
        response = WriterSamplesResponse(
            success=True,
            writer_number=writer_number,
            samples=samples,
            count=len(samples)
        )
        
        logger.info(f"Retrieved {len(samples)} samples for writer: {writer_number}")
        return response
        
    except Exception as e:
        logger.error(f"Error getting writer samples: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.put("/samples/{sample_id}/scores", response_model=UpdateScoresResponse, summary="評価スコア更新")
async def update_sample_scores(sample_id: int, request: UpdateScoresRequest):
    """
    指定したサンプルの評価スコアを更新
    
    Args:
        sample_id: サンプルID
        request: 更新リクエスト（スコア、コメント、評価者）
        
    Returns:
        更新結果
    """
    if supabase_processor is None:
        raise HTTPException(status_code=503, detail="Supabase processor not initialized")
    
    try:
        updated_data = supabase_processor.update_sample_scores(
            sample_id=sample_id,
            scores=request.scores,
            comments=request.comments,
            evaluator=request.evaluator
        )
        
        response = UpdateScoresResponse(
            success=True,
            message=f"Sample {sample_id} scores updated successfully",
            sample_id=sample_id,
            updated_data=updated_data
        )
        
        logger.info(f"Updated scores for sample ID: {sample_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error updating sample scores: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats", response_model=StatsResponse, summary="統計情報取得")
async def get_stats():
    """
    データベース統計情報を取得
    
    Returns:
        統計情報（記入者数、文字数、サンプル数など）
    """
    if supabase_processor is None:
        raise HTTPException(status_code=503, detail="Supabase processor not initialized")
    
    try:
        stats = supabase_processor.get_database_stats()
        
        response = StatsResponse(
            success=True,
            stats=stats,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info("Retrieved database statistics")
        return response
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/ml-dataset", summary="機械学習用データセット取得")
async def get_ml_dataset(quality_status: str = "approved"):
    """
    機械学習用データセットを取得
    
    Args:
        quality_status: 品質ステータス（approved/pending/rejected）
        
    Returns:
        機械学習用データセット
    """
    if supabase_processor is None:
        raise HTTPException(status_code=503, detail="Supabase processor not initialized")
    
    try:
        dataset = supabase_processor.get_ml_dataset(quality_status)
        
        return {
            "success": True,
            "quality_status": quality_status,
            "dataset": dataset,
            "count": len(dataset),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting ML dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===========================
# エラーハンドラー
# ===========================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "/docs - API Documentation",
                "/health - Health Check",
                "/process-form - Form Processing",
                "/samples/{writer_number} - Get Writer Samples",
                "/stats - Get Statistics"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# ===========================
# メイン実行関数
# ===========================

def main():
    """
    開発サーバー起動
    """
    # 環境変数確認
    required_env_vars = ['SUPABASE_URL', 'SUPABASE_KEY', 'GEMINI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please check your .env file or environment settings")
        return
    
    # サーバー起動
    uvicorn.run(
        "supabase_api_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()