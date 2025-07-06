#!/usr/bin/env python3
"""
API サーバー起動スクリプト
- 環境変数チェック
- 依存関係確認
- サーバー起動
Created: 2025-01-06
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """環境変数チェック"""
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_KEY', 
        'GEMINI_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please set the following environment variables:")
        for var in missing_vars:
            logger.info(f"  export {var}=your_value_here")
        return False
    
    logger.info("All required environment variables are set")
    return True

def check_dependencies():
    """依存関係チェック"""
    try:
        import fastapi
        import uvicorn
        import supabase
        logger.info("All required packages are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Please install dependencies:")
        logger.info("  pip install -r requirements_api.txt")
        return False

def load_env_file():
    """環境変数ファイル読み込み"""
    env_file = Path('.env')
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("Environment variables loaded from .env file")
        except ImportError:
            logger.warning("python-dotenv not installed, skipping .env file")
    else:
        logger.info(".env file not found, using system environment variables")

def start_development_server():
    """開発サーバー起動"""
    logger.info("Starting development server...")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "supabase_api_server:app",
        "--host", "0.0.0.0",
        "--port", "8001",
        "--reload",
        "--log-level", "info"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Server failed to start: {e}")
        return False
    
    return True

def start_production_server():
    """本番サーバー起動"""
    logger.info("Starting production server...")
    
    port = os.getenv('PORT', '8001')
    workers = os.getenv('WORKERS', '1')
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "supabase_api_server:app",
        "--host", "0.0.0.0",
        "--port", port,
        "--workers", workers,
        "--log-level", "info"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Server failed to start: {e}")
        return False
    
    return True

def main():
    """メイン実行"""
    logger.info("=== 手書き文字評価 API サーバー起動 ===")
    
    # 環境変数ファイル読み込み
    load_env_file()
    
    # 環境チェック
    if not check_environment():
        logger.error("Environment check failed")
        sys.exit(1)
    
    # 依存関係チェック
    if not check_dependencies():
        logger.error("Dependency check failed")
        sys.exit(1)
    
    # サーバー起動
    is_production = os.getenv('RENDER') or os.getenv('PRODUCTION')
    
    if is_production:
        logger.info("Running in production mode")
        success = start_production_server()
    else:
        logger.info("Running in development mode")
        success = start_development_server()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()