"""
test_integration.py
==================
実際のサンプル画像を使った統合テスト
"""

import pytest
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.preprocessing import preprocess
from src.eval.pipeline import evaluate_pair, evaluate_all
from src.eval.cli import main


class TestRealImageIntegration:
    """実際の画像ファイルを使った統合テスト"""
    
    def test_sample_images_exist(self, test_data_dir):
        """サンプル画像が存在することを確認"""
        samples_dir = test_data_dir / "samples"
        
        # お手本画像が存在することを確認
        ref_images = list(samples_dir.glob("ref_*.jpg"))
        assert len(ref_images) > 0, "お手本画像が見つかりません"
        
        # ユーザー画像が存在することを確認
        user_images = list(samples_dir.glob("user_*.jpg"))
        assert len(user_images) > 0, "ユーザー画像が見つかりません"
    
    def test_preprocess_real_images(self, test_data_dir):
        """実際の画像での前処理テスト"""
        samples_dir = test_data_dir / "samples"
        
        ref_images = list(samples_dir.glob("ref_*.jpg"))
        user_images = list(samples_dir.glob("user_*.jpg"))
        
        if ref_images and user_images:
            ref_path = ref_images[0]
            user_path = user_images[0]
            
            # 前処理を実行
            ref_img, ref_mask = preprocess(ref_path, size=256, dbg=False)
            user_img, user_mask = preprocess(user_path, size=256, dbg=False)
            
            # 結果の検証
            assert ref_img.shape == (256, 256)
            assert ref_mask.shape == (256, 256)
            assert user_img.shape == (256, 256)
            assert user_mask.shape == (256, 256)
            
            # マスクが空でないことを確認
            assert ref_mask.sum() > 0, "お手本マスクが空です"
            assert user_mask.sum() > 0, "ユーザーマスクが空です"
    
    def test_evaluate_real_images(self, test_data_dir):
        """実際の画像での評価テスト"""
        samples_dir = test_data_dir / "samples"
        
        ref_images = list(samples_dir.glob("ref_*.jpg"))
        user_images = list(samples_dir.glob("user_*.jpg"))
        
        if ref_images and user_images:
            ref_path = ref_images[0]
            user_path = user_images[0]
            
            ref_img, ref_mask = preprocess(ref_path, size=256, dbg=False)
            user_img, user_mask = preprocess(user_path, size=256, dbg=False)
            
            scores = evaluate_pair(ref_img, ref_mask, user_img, user_mask)
            
            # スコアの検証
            assert "形" in scores
            assert "黒" in scores
            assert "白" in scores
            assert "場" in scores
            assert "total" in scores
            
            # スコアが有効な範囲内であることを確認
            for key in ["形", "黒", "白", "場", "total"]:
                assert 0 <= scores[key] <= 100, f"{key}スコアが範囲外: {scores[key]}"
    
    def test_evaluate_all_function(self, test_data_dir):
        """evaluate_all関数のテスト"""
        samples_dir = test_data_dir / "samples"
        
        ref_images = list(samples_dir.glob("ref_*.jpg"))
        user_images = list(samples_dir.glob("user_*.jpg"))
        
        if ref_images and user_images:
            ref_path = ref_images[0]
            
            results = evaluate_all(ref_path, user_images, size=256, dbg=False)
            
            assert len(results) == len(user_images)
            
            for result in results:
                assert "file" in result
                if "scores" in result:
                    scores = result["scores"]
                    for key in ["形", "黒", "白", "場", "total"]:
                        assert 0 <= scores[key] <= 100
                elif "error" in result:
                    # エラーが発生した場合でも適切にハンドリングされていることを確認
                    assert isinstance(result["error"], str)
    
    def test_cli_with_real_images(self, test_data_dir):
        """実際の画像でのCLI動作テスト"""
        samples_dir = test_data_dir / "samples"
        
        ref_images = list(samples_dir.glob("ref_*.jpg"))
        user_images = list(samples_dir.glob("user_*.jpg"))
        
        if ref_images and user_images:
            import sys
            from io import StringIO
            
            ref_path = ref_images[0]
            user_path = user_images[0]
            
            # 引数を模擬
            original_argv = sys.argv
            sys.argv = ['cli.py', str(ref_path), str(user_path)]
            
            # 標準出力をキャプチャ
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                # CLIメイン関数を実行
                result = main()
                output = captured_output.getvalue()
                
                # 正常終了することを確認
                assert result == 0
                
                # 出力が適切な形式であることを確認
                assert "形" in output
                assert "黒" in output
                assert "白" in output
                assert "場" in output
                assert "total" in output
                
            finally:
                sys.stdout = old_stdout
                sys.argv = original_argv


@pytest.mark.parametrize("size", [128, 256, 512])
def test_different_sizes(test_data_dir, size):
    """異なるサイズでの前処理テスト"""
    samples_dir = test_data_dir / "samples"
    ref_images = list(samples_dir.glob("ref_*.jpg"))
    
    if ref_images:
        ref_path = ref_images[0]
        
        try:
            ref_img, ref_mask = preprocess(ref_path, size=size, dbg=False)
            assert ref_img.shape == (size, size)
            assert ref_mask.shape == (size, size)
        except Exception as e:
            pytest.fail(f"Size {size} でテストが失敗: {e}")


def test_performance_benchmark(test_data_dir):
    """パフォーマンスベンチマークテスト"""
    import time
    
    samples_dir = test_data_dir / "samples"
    ref_images = list(samples_dir.glob("ref_*.jpg"))
    user_images = list(samples_dir.glob("user_*.jpg"))
    
    if ref_images and user_images:
        ref_path = ref_images[0]
        user_path = user_images[0]
        
        # 前処理のベンチマーク
        start_time = time.time()
        ref_img, ref_mask = preprocess(ref_path, size=256, dbg=False)
        user_img, user_mask = preprocess(user_path, size=256, dbg=False)
        preprocess_time = time.time() - start_time
        
        # 評価のベンチマーク
        start_time = time.time()
        scores = evaluate_pair(ref_img, ref_mask, user_img, user_mask)
        evaluate_time = time.time() - start_time
        
        # パフォーマンス要件（これらは調整可能）
        assert preprocess_time < 5.0, f"前処理が遅すぎます: {preprocess_time:.2f}秒"
        assert evaluate_time < 1.0, f"評価が遅すぎます: {evaluate_time:.2f}秒"
        
        print(f"前処理時間: {preprocess_time:.3f}秒")
        print(f"評価時間: {evaluate_time:.3f}秒")
