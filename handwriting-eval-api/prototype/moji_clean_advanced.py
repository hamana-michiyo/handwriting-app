"""
改良版補助線除去プログラム
複数の高度な手法を実装・比較
"""
import cv2
import numpy as np
import os

def remove_guidelines_by_components(img):
    """連結成分解析で補助線を選択的に除去"""
    # アダプティブ二値化（画像に応じた最適な閾値）
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 連結成分解析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    result = np.zeros_like(binary)
    removed_components = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # 補助線判定（細長い形状を除去）
        aspect_ratio = max(width, height) / (min(width, height) + 1)
        
        # 文字成分を保持（適度なアスペクト比 + 十分な面積）
        if aspect_ratio < 15 and area > 20:
            result[labels == i] = 255
        else:
            removed_components.append((i, area, width, height, aspect_ratio))
    
    print(f"連結成分解析: {len(removed_components)}個の補助線成分を除去")
    for comp in removed_components[:5]:  # 最初の5つを表示
        print(f"  成分{comp[0]}: 面積={comp[1]}, サイズ={comp[2]}x{comp[3]}, アスペクト比={comp[4]:.1f}")
    
    return cv2.bitwise_not(result)

def remove_guidelines_directional(img):
    """方向別カーネルで補助線を除去"""
    # アダプティブ二値化
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 水平・垂直カーネル（補助線の幅に応じて調整）
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    
    # 補助線検出
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    guidelines = cv2.bitwise_or(h_lines, v_lines)
    
    # 検出された補助線の統計
    h_pixels = cv2.countNonZero(h_lines)
    v_pixels = cv2.countNonZero(v_lines)
    print(f"方向別検出: 水平線={h_pixels}px, 垂直線={v_pixels}px")
    
    # 補助線を元画像から除去
    result = cv2.bitwise_and(binary, cv2.bitwise_not(guidelines))
    return cv2.bitwise_not(result)

def remove_guidelines_advanced(img):
    """統合手法による高精度補助線除去"""
    # 1. Otsu二値化（自動最適閾値）
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. 連結成分解析で補助線検出
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(otsu)
    
    # 補助線マスク作成
    guideline_mask = np.zeros_like(otsu)
    removed_count = 0
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH] 
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # 細長い成分を補助線と判定
        aspect_ratio = max(width, height) / (min(width, height) + 1)
        
        # より厳密な補助線判定
        is_guideline = (
            (aspect_ratio > 10 and area < 500) or  # 細長い
            (width > img.shape[1] * 0.8) or        # 画像幅の80%以上
            (height > img.shape[0] * 0.8)          # 画像高さの80%以上
        )
        
        if is_guideline:
            guideline_mask[labels == i] = 255
            removed_count += 1
    
    print(f"統合手法: {removed_count}個の補助線成分を除去")
    
    # 3. 補助線領域を膨張して完全除去
    dilated_mask = cv2.dilate(guideline_mask, np.ones((3,3), np.uint8), iterations=1)
    
    # 4. 元画像から補助線を除去
    result = cv2.bitwise_and(otsu, cv2.bitwise_not(dilated_mask))
    
    # 5. 軽いモルフォロジーでクリーンアップ
    kernel = np.ones((2,2), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    return cv2.bitwise_not(result)

def original_method(img):
    """元の手法（比較用）"""
    # ガウシアン
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 二値化
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    
    # モルフォロジー（ちょい弱め）
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 反転して背景白・文字黒
    result = cv2.bitwise_not(opening)
    
    return result

def main():
    """複数手法の比較テスト"""
    input_path = "/workspace/result/chars/char_1.png"
    
    # 画像読み込み
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {input_path}")
    
    print(f"元画像サイズ: {img.shape}")
    print("=" * 50)
    
    # 各手法を適用
    methods = {
        "original": ("元手法", original_method),
        "components": ("連結成分解析", remove_guidelines_by_components),
        "directional": ("方向別フィルタ", remove_guidelines_directional),
        "advanced": ("統合手法", remove_guidelines_advanced)
    }
    
    results = {}
    for method_key, (method_name, method_func) in methods.items():
        print(f"\n[{method_name}]")
        result = method_func(img)
        results[method_key] = result
        
        # 結果保存
        output_path = f"/workspace/result/chars/char_1_{method_key}_removed.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"保存: {output_path}")
    
    # 比較統計
    print("\n" + "=" * 50)
    print("結果比較")
    print("=" * 50)
    
    for method_key, (method_name, _) in methods.items():
        result = results[method_key]
        white_pixels = np.sum(result == 255)  # 背景（白）
        black_pixels = np.sum(result == 0)    # 文字（黒）
        total_pixels = result.size
        
        text_ratio = black_pixels / total_pixels * 100
        print(f"{method_name:12}: 文字領域={text_ratio:.1f}% ({black_pixels}/{total_pixels})")
    
    print(f"\n全ての結果が result/chars/ に保存されました")
    print("手法比較:")
    print("- original: 元の手法（ガウシアン + 固定閾値）")
    print("- components: 連結成分解析（形状による判定）")
    print("- directional: 方向別フィルタ（水平・垂直線除去）")
    print("- advanced: 統合手法（複数手法の組み合わせ）")

if __name__ == "__main__":
    main()