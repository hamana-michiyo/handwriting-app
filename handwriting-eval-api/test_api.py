#!/usr/bin/env python3
"""
FastAPI サーバー テストスクリプト
- エンドポイント動作確認
- Supabase統合テスト
- サンプルリクエスト実行
Created: 2025-01-06
"""

import os
import sys
import json
import base64
import requests
import time
from pathlib import Path

# ===========================
# 設定
# ===========================

API_BASE_URL = "http://localhost:8001"
TEST_IMAGE_PATH = "docs/記入sample.JPG"

# ===========================
# テスト関数
# ===========================

def test_health_check():
    """ヘルスチェックテスト"""
    print("=== Health Check Test ===")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data.get('status')}")
            print(f"Database: {data.get('database_status')}")
            print(f"Stats: {data.get('stats')}")
            return True
        else:
            print(f"Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def test_api_info():
    """API情報テスト"""
    print("\n=== API Info Test ===")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"API Name: {data.get('name')}")
            print(f"Version: {data.get('version')}")
            print(f"Features: {len(data.get('features', []))}")
            return True
        else:
            print(f"API info failed: {response.text}")
            return False
    except Exception as e:
        print(f"API info error: {e}")
        return False

def test_stats():
    """統計情報テスト"""
    print("\n=== Stats Test ===")
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            stats = data.get('stats', {})
            print(f"Writers: {stats.get('writers_count')}")
            print(f"Characters: {stats.get('characters_count')}")
            print(f"Samples: {stats.get('samples_count')}")
            print(f"Approval Rate: {stats.get('approval_rate', 0):.1f}%")
            return True
        else:
            print(f"Stats failed: {response.text}")
            return False
    except Exception as e:
        print(f"Stats error: {e}")
        return False

def test_process_form_base64():
    """記入用紙処理テスト（Base64）"""
    print("\n=== Form Processing Test (Base64) ===")
    
    # テスト画像確認
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Test image not found: {TEST_IMAGE_PATH}")
        return False
    
    try:
        # 画像をBase64エンコード
        with open(TEST_IMAGE_PATH, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # リクエストデータ
        request_data = {
            "image_base64": image_base64,
            "writer_number": "api_test_writer",
            "writer_age": 25,
            "writer_grade": "テスト",
            "auto_save": True
        }
        
        print(f"Sending request with image size: {len(image_data)} bytes")
        
        # API呼び出し
        response = requests.post(
            f"{API_BASE_URL}/process-form",
            json=request_data,
            timeout=60  # Gemini API処理時間を考慮
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Message: {data.get('message')}")
            print(f"Character Results: {len(data.get('character_results', []))}")
            print(f"Number Results: {len(data.get('number_results', []))}")
            
            # 文字認識結果詳細
            char_results = data.get('character_results', [])
            for i, result in enumerate(char_results):
                if result.get('gemini_result'):
                    gemini = result['gemini_result']
                    print(f"  Character {i+1}: {gemini.get('character')} ({gemini.get('confidence', 0)*100:.1f}%)")
                    if result.get('saved_to_supabase'):
                        print(f"    Saved to DB: ID={result.get('sample_id')}")
                        if result.get('is_duplicate'):
                            print(f"    Action: SKIPPED (duplicate)")
                        else:
                            print(f"    Action: CREATED (new)")
            
            return True
        else:
            print(f"Form processing failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"Form processing error: {e}")
        return False

def test_get_writer_samples():
    """記入者サンプル取得テスト"""
    print("\n=== Writer Samples Test ===")
    
    writer_number = "api_test_writer"
    
    try:
        response = requests.get(f"{API_BASE_URL}/samples/{writer_number}")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Writer: {data.get('writer_number')}")
            print(f"Sample Count: {data.get('count')}")
            
            samples = data.get('samples', [])
            for sample in samples[:3]:  # 最初の3件のみ表示
                print(f"  Sample ID: {sample.get('id')}")
                print(f"    Character: {sample.get('characters', {}).get('character')}")
                print(f"    Gemini: {sample.get('gemini_recognized_char')} ({sample.get('gemini_confidence', 0)*100:.1f}%)")
                print(f"    Quality: {sample.get('quality_status')}")
            
            return True
        else:
            print(f"Writer samples failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"Writer samples error: {e}")
        return False

def test_ml_dataset():
    """機械学習データセットテスト"""
    print("\n=== ML Dataset Test ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/ml-dataset?quality_status=pending")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Quality Status: {data.get('quality_status')}")
            print(f"Dataset Count: {data.get('count')}")
            return True
        else:
            print(f"ML dataset failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"ML dataset error: {e}")
        return False

def main():
    """メインテスト実行"""
    print("=== FastAPI Supabase統合 テストスクリプト ===")
    print(f"API Base URL: {API_BASE_URL}")
    
    # サーバー起動確認
    print("Checking server availability...")
    for i in range(5):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is available")
                break
            else:
                print(f"❌ Server returned {response.status_code}")
        except Exception as e:
            print(f"⏳ Waiting for server... ({i+1}/5)")
            time.sleep(2)
    else:
        print("❌ Server is not available. Please start the API server first:")
        print("  python start_api.py")
        sys.exit(1)
    
    # テスト実行
    tests = [
        ("Health Check", test_health_check),
        ("API Info", test_api_info),
        ("Stats", test_stats),
        ("Form Processing", test_process_form_base64),
        ("Writer Samples", test_get_writer_samples),
        ("ML Dataset", test_ml_dataset)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        result = test_func()
        results.append((test_name, result))
        print(f"Result: {'✅ PASS' if result else '❌ FAIL'}")
    
    # 結果サマリー
    print(f"\n{'='*50}")
    print("=== Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()