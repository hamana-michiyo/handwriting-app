"""
metrics.py
==========
手書き文字評価の4軸（形・黒・白・場）のスコア計算関数
"""

import cv2
import numpy as np

# 計算効率化のためのキャッシュ（Phase 1 最適化）
_distance_transform_cache = {}
_analysis_cache = {}

def _get_distance_transform_cached(mask):
    """距離変換結果のキャッシュ機能（計算効率化）"""
    mask_hash = hash(mask.tobytes())
    if mask_hash not in _distance_transform_cache:
        _distance_transform_cache[mask_hash] = cv2.distanceTransform(
            mask.astype(np.uint8), cv2.DIST_L2, 5
        )
    return _distance_transform_cache[mask_hash]

def _clear_cache():
    """キャッシュクリア（メモリ管理）"""
    global _distance_transform_cache, _analysis_cache
    _distance_transform_cache.clear()
    _analysis_cache.clear()

def hu_moment_similarity(mask_ref, mask_user):
    """
    Huモーメントによる形状類似度を計算する
    
    Args:
        mask_ref: お手本の二値化マスク
        mask_user: ユーザーの二値化マスク
        
    Returns:
        float: Huモーメント類似度（0-1）、1に近いほど類似
    """
    # 輪郭抽出
    contours_ref, _ = cv2.findContours(mask_ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_user, _ = cv2.findContours(mask_user.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours_ref) == 0 or len(contours_user) == 0:
        return 0.0
    
    # 最大の輪郭を使用
    ref_contour = max(contours_ref, key=cv2.contourArea)
    user_contour = max(contours_user, key=cv2.contourArea)
    
    # 面積チェック
    if cv2.contourArea(ref_contour) < 10 or cv2.contourArea(user_contour) < 10:
        return 0.0
    
    # Huモーメント計算
    hu_ref = cv2.HuMoments(cv2.moments(ref_contour)).flatten()
    hu_user = cv2.HuMoments(cv2.moments(user_contour)).flatten()
    
    # 対数変換（ゼロ除算対策）
    hu_ref = np.sign(hu_ref) * np.log10(np.abs(hu_ref) + 1e-10)
    hu_user = np.sign(hu_user) * np.log10(np.abs(hu_user) + 1e-10)
    
    # ユークリッド距離を計算し、類似度に変換
    distance = np.linalg.norm(hu_ref - hu_user)
    similarity = np.exp(-distance / 10.0)  # 適度なスケーリング
    
    return float(similarity)


def shape_score(mask_ref, mask_user):
    """
    形スコア（ハイブリッド評価：位置補正IoU + Huモーメント + スケール補正）を計算する
    
    位置ずれ・サイズ違いに対してロバストな形状評価を行う：
    1. テンプレートマッチングで最適な重ね合わせ位置を特定
    2. スケール正規化による相似形対応
    3. その位置・スケールでのIoUを計算
    4. 改良Huモーメントによる形状類似度を計算
    5. 重み付き統合スコア
    
    Args:
        mask_ref: お手本の二値化マスク
        mask_user: ユーザーの二値化マスク
        
    Returns:
        float: スケール対応ハイブリッド形状スコア（0-1）
    """
    # 空のマスクチェック
    if mask_ref.sum() == 0 or mask_user.sum() == 0:
        return 0.0
    
    # 1. スケール正規化位置補正IoUを計算
    scale_corrected_iou = _calculate_scale_corrected_iou(mask_ref, mask_user)
    
    # 2. 改良Huモーメント類似度を計算
    hu_similarity = _improved_hu_moment_similarity(mask_ref, mask_user)
    
    # 3. 重み付き統合（IoU重視、Huモーメントで補完）
    iou_weight = 0.7
    hu_weight = 0.3
    
    hybrid_score = iou_weight * scale_corrected_iou + hu_weight * hu_similarity
    
    return float(hybrid_score)


def _calculate_position_corrected_iou(mask_ref, mask_user):
    """
    位置補正IoUを計算する内部関数
    
    Args:
        mask_ref: お手本の二値化マスク
        mask_user: ユーザーの二値化マスク
        
    Returns:
        float: 位置補正済みIoU（0-1）
    """
    
    h_ref, w_ref = mask_ref.shape
    h_user, w_user = mask_user.shape
    
    # サイズが同じ場合はテンプレートマッチングを使用
    if h_ref == h_user and w_ref == w_user:
        # テンプレートマッチングで最適位置を探索
        mask_ref_uint8 = mask_ref.astype(np.uint8) * 255
        mask_user_uint8 = mask_user.astype(np.uint8) * 255
        
        # パディングして検索範囲を拡張
        pad_size = max(h_user, w_user) // 2
        padded_ref = np.pad(mask_ref_uint8, pad_size, mode='constant', constant_values=0)
        
        # テンプレートマッチング実行
        result = cv2.matchTemplate(padded_ref, mask_user_uint8, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # 最適位置でユーザーマスクを配置
        corrected_user = np.zeros_like(mask_ref, dtype=bool)
        
        # パディングを考慮した位置計算
        top = max_loc[1] - pad_size
        left = max_loc[0] - pad_size
        bottom = min(top + h_user, h_ref)
        right = min(left + w_user, w_ref)
        
        # 範囲チェック
        if top < h_ref and left < w_ref and bottom > 0 and right > 0:
            # クリッピング
            ref_top = max(0, top)
            ref_left = max(0, left)
            ref_bottom = min(bottom, h_ref)
            ref_right = min(right, w_ref)
            
            # ユーザーマスクの対応範囲
            user_top = max(0, -top)
            user_left = max(0, -left)
            user_bottom = user_top + (ref_bottom - ref_top)
            user_right = user_left + (ref_right - ref_left)
            
            # 位置補正済みマスクに配置
            corrected_user[ref_top:ref_bottom, ref_left:ref_right] = \
                mask_user[user_top:user_bottom, user_left:user_right]
        
        # 位置補正後のIoU計算
        union = np.logical_or(mask_ref, corrected_user).sum()
        inter = np.logical_and(mask_ref, corrected_user).sum()
        
        if union == 0:
            return 0.0
        
        iou = float(inter / union)
        return iou
    
    else:
        # サイズが異なる場合は中央揃えでIoU計算
        # より大きな共通キャンバスを作成
        max_h = max(h_ref, h_user)
        max_w = max(w_ref, w_user)
        
        # 中央揃えで配置
        ref_padded = np.zeros((max_h, max_w), dtype=bool)
        user_padded = np.zeros((max_h, max_w), dtype=bool)
        
        ref_start_h = (max_h - h_ref) // 2
        ref_start_w = (max_w - w_ref) // 2
        user_start_h = (max_h - h_user) // 2
        user_start_w = (max_w - w_user) // 2
        
        ref_padded[ref_start_h:ref_start_h + h_ref, ref_start_w:ref_start_w + w_ref] = mask_ref
        user_padded[user_start_h:user_start_h + h_user, user_start_w:user_start_w + w_user] = mask_user
        
        # IoU計算
        union = np.logical_or(ref_padded, user_padded).sum()
        inter = np.logical_and(ref_padded, user_padded).sum()
        
        if union == 0:
            return 0.0
        
        iou = float(inter / union)
        return iou


def _calculate_scale_corrected_iou(mask_ref, mask_user):
    """
    スケール正規化位置補正IoUを計算する
    
    相似形を適切に評価するため、最適なスケールと位置を探索してIoUを計算
    
    Args:
        mask_ref: お手本の二値化マスク
        mask_user: ユーザーの二値化マスク
        
    Returns:
        float: スケール・位置補正済みIoU（0-1）
    """
    # 基本的な位置補正IoU
    position_iou = _calculate_position_corrected_iou(mask_ref, mask_user)
    
    # スケール補正も試行
    best_iou = position_iou
    
    # 複数のスケールファクターを試行
    scale_factors = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]
    
    for scale in scale_factors:
        if scale == 1.0:
            continue  # 既に計算済み
            
        # ユーザーマスクをスケール変換
        scaled_user = _scale_mask(mask_user, scale)
        
        if scaled_user.sum() == 0:
            continue
            
        # スケール後の位置補正IoU計算
        scaled_iou = _calculate_position_corrected_iou(mask_ref, scaled_user)
        
        if scaled_iou > best_iou:
            best_iou = scaled_iou
    
    return best_iou


def _scale_mask(mask, scale_factor):
    """
    マスクを指定倍率でスケール変換する
    
    Args:
        mask: 入力マスク
        scale_factor: スケール倍率
        
    Returns:
        numpy.ndarray: スケール変換されたマスク
    """
    if scale_factor == 1.0:
        return mask
    
    h, w = mask.shape
    
    # OpenCVでリサイズ
    mask_uint8 = mask.astype(np.uint8) * 255
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    if new_h <= 0 or new_w <= 0:
        return np.zeros_like(mask, dtype=bool)
    
    # リサイズ実行
    resized = cv2.resize(mask_uint8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # 元のサイズに合わせて中央配置
    result = np.zeros_like(mask, dtype=bool)
    
    # 中央配置の計算
    start_h = max(0, (h - new_h) // 2)
    start_w = max(0, (w - new_w) // 2)
    end_h = min(h, start_h + new_h)
    end_w = min(w, start_w + new_w)
    
    # リサイズ画像の対応範囲
    src_start_h = max(0, (new_h - h) // 2)
    src_start_w = max(0, (new_w - w) // 2)
    src_end_h = src_start_h + (end_h - start_h)
    src_end_w = src_start_w + (end_w - start_w)
    
    # 配置
    result[start_h:end_h, start_w:end_w] = resized[src_start_h:src_end_h, src_start_w:src_end_w] > 127
    
    return result


def _improved_hu_moment_similarity(mask_ref, mask_user):
    """
    改良Huモーメント類似度を計算する（サイズ正規化対応）
    
    Args:
        mask_ref: お手本の二値化マスク
        mask_user: ユーザーの二値化マスク
        
    Returns:
        float: 改良Huモーメント類似度（0-1）
    """
    # 基本的な形状記述子による評価を併用
    basic_similarity = _basic_shape_similarity(mask_ref, mask_user)
    
    # 輪郭抽出
    contours_ref, _ = cv2.findContours(mask_ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_user, _ = cv2.findContours(mask_user.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours_ref) == 0 or len(contours_user) == 0:
        return basic_similarity
    
    # 最大の輪郭を使用
    ref_contour = max(contours_ref, key=cv2.contourArea)
    user_contour = max(contours_user, key=cv2.contourArea)
    
    # 面積チェック
    if cv2.contourArea(ref_contour) < 5 or cv2.contourArea(user_contour) < 5:
        return basic_similarity
    
    try:
        # OpenCVのmatchShapesを使用（より安定）
        distance = cv2.matchShapes(ref_contour, user_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        
        # matchShapesは距離なので、類似度に変換
        # 距離が0に近いほど類似
        similarity = np.exp(-distance * 2.0)  # スケーリング調整
        
        # 基本形状記述子と組み合わせ
        combined_similarity = 0.6 * similarity + 0.4 * basic_similarity
        
        return float(combined_similarity)
        
    except:
        # エラーの場合は基本形状記述子のみ
        return basic_similarity


def _basic_shape_similarity(mask_ref, mask_user):
    """
    基本的な形状記述子による類似度計算
    
    Args:
        mask_ref: お手本の二値化マスク
        mask_user: ユーザーの二値化マスク
        
    Returns:
        float: 基本形状類似度（0-1）
    """
    # 輪郭抽出
    contours_ref, _ = cv2.findContours(mask_ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_user, _ = cv2.findContours(mask_user.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours_ref) == 0 or len(contours_user) == 0:
        return 0.0
    
    # 最大の輪郭を使用
    ref_contour = max(contours_ref, key=cv2.contourArea)
    user_contour = max(contours_user, key=cv2.contourArea)
    
    # 基本的な形状記述子を計算
    def get_shape_descriptors(contour):
        area = cv2.contourArea(contour)
        if area < 5:
            return None
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None
            
        # 円形度 = 4π * 面積 / 周囲長^2
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 凸包面積比
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # バウンディングボックスのアスペクト比
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        return [circularity, convexity, aspect_ratio]
    
    ref_desc = get_shape_descriptors(ref_contour)
    user_desc = get_shape_descriptors(user_contour)
    
    if ref_desc is None or user_desc is None:
        return 0.0
    
    # 各記述子の類似度を計算
    similarities = []
    for r, u in zip(ref_desc, user_desc):
        if r == 0 and u == 0:
            sim = 1.0
        else:
            # 相対誤差ベースの類似度
            diff = abs(r - u) / (max(r, u) + 1e-6)
            sim = np.exp(-diff * 3.0)
        similarities.append(sim)
    
    # 平均類似度
    return float(np.mean(similarities))


def stroke_cv(mask):
    """
    線幅の変動係数を計算する（黒スコア用）
    
    Args:
        mask: 二値化マスク
        
    Returns:
        float or None: 変動係数（CV）、計算できない場合はNone
    """
    # 距離変換で各点の半径を推定 -> 線幅=半径*2
    dist = _get_distance_transform_cached(mask)
    widths = dist[mask > 0] * 2
    widths = widths[widths > 0.5]          # 端の0を除外
    if len(widths) < 20:                   # 点が少なすぎ
        return None
    return widths.std() / widths.mean()    # 変動係数 (CV)


def black_score(cv_ref, cv_user):
    """
    黒スコア（線幅ばらつき差）を計算する
    
    Args:
        cv_ref: お手本の変動係数
        cv_user: ユーザーの変動係数
        
    Returns:
        float: 黒スコア（0-1）
    """
    if cv_ref is None or cv_user is None:
        return 0.0
    sigma = cv_ref * 0.5 + 1e-6
    return np.exp(-((cv_user - cv_ref) ** 2) / (2 * sigma ** 2))


def black_ratio(mask):
    """
    黒画素の割合を計算する
    
    Args:
        mask: 二値化マスク
        
    Returns:
        float: 黒画素割合（0-1）
    """
    return mask.mean()   # 0–1


def white_score(r_ref, r_user):
    """
    白スコア（黒画素割合差）を計算する
    
    Args:
        r_ref: お手本の黒画素割合
        r_user: ユーザーの黒画素割合
        
    Returns:
        float: 白スコア（0-1）
    """
    sigma = r_ref * 0.5 + 1e-6
    return np.exp(-((r_user - r_ref) ** 2) / (2 * sigma ** 2))


def center_score(mask):
    """
    場スコア（重心位置）を計算する
    
    Args:
        mask: 二値化マスク
        
    Returns:
        float: 場スコア（0-1）、中央に近いほど高い
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    cy, cx = ys.mean(), xs.mean()
    dist = np.hypot((cx - w/2)/(w/2),
                    (cy - h/2)/(h/2))
    return 1.0 - min(dist, 1.0)


def analyze_stroke_intensity(gray_image, mask):
    """
    線の濃淡均一性を解析する（黒スコア用）
    
    Args:
        gray_image: グレースケール画像（0-255）
        mask: 二値化マスク
        
    Returns:
        dict: 濃淡解析結果
            - mean_intensity: 平均濃度
            - intensity_std: 濃度標準偏差
            - intensity_cv: 濃度変動係数
            - uniformity_score: 均一性スコア（0-1）
    """
    if mask.sum() == 0:
        return {
            'mean_intensity': 0.0,
            'intensity_std': 0.0,
            'intensity_cv': 0.0,
            'uniformity_score': 0.0
        }
    
    # マスク領域の濃度値を取得（反転: 黒い部分=高濃度として扱う）
    stroke_pixels = 255 - gray_image[mask > 0]  # 白背景・黒文字を前提に反転
    
    mean_intensity = np.mean(stroke_pixels)
    intensity_std = np.std(stroke_pixels)
    
    # 変動係数（CV）
    intensity_cv = intensity_std / (mean_intensity + 1e-6)
    
    # 均一性スコア：CVが小さいほど高い（理想的な線は濃度が均一）
    # 薄すぎる線（平均濃度が低い）もペナルティ
    intensity_factor = np.clip(mean_intensity / 128.0, 0.3, 1.0)  # 薄すぎ補正
    cv_factor = np.exp(-intensity_cv * 3.0)  # CVペナルティ
    uniformity_score = intensity_factor * cv_factor
    
    return {
        'mean_intensity': float(mean_intensity),
        'intensity_std': float(intensity_std),
        'intensity_cv': float(intensity_cv),
        'uniformity_score': float(uniformity_score)
    }


def comprehensive_black_score(gray_ref, mask_ref, gray_user, mask_user):
    """
    包括的な黒スコア（線幅安定性 + 濃淡均一性）を計算する
    
    Args:
        gray_ref: お手本のグレースケール画像
        mask_ref: お手本の二値化マスク
        gray_user: ユーザーのグレースケール画像
        mask_user: ユーザーの二値化マスク
        
    Returns:
        dict: 詳細な黒スコア解析結果
    """
    # 1. 線幅安定性（既存）
    cv_ref = stroke_cv(mask_ref)
    cv_user = stroke_cv(mask_user)
    width_stability_score = black_score(cv_ref, cv_user)
    
    # 2. 濃淡均一性（新規）
    intensity_ref = analyze_stroke_intensity(gray_ref, mask_ref)
    intensity_user = analyze_stroke_intensity(gray_user, mask_user)
    
    # 濃淡の類似度評価
    if intensity_ref['uniformity_score'] == 0.0 and intensity_user['uniformity_score'] == 0.0:
        intensity_similarity = 1.0
    else:
        # 均一性スコアの差を評価
        uniformity_diff = abs(intensity_ref['uniformity_score'] - intensity_user['uniformity_score'])
        sigma = intensity_ref['uniformity_score'] * 0.3 + 1e-6
        intensity_similarity = np.exp(-uniformity_diff**2 / (2 * sigma**2))
    
    # 3. 統合スコア
    width_weight = 0.6  # 線幅安定性の重み
    intensity_weight = 0.4  # 濃淡均一性の重み
    
    total_black_score = width_weight * width_stability_score + intensity_weight * intensity_similarity
    
    return {
        'width_stability': float(width_stability_score),
        'intensity_similarity': float(intensity_similarity),
        'total_score': float(total_black_score),
        'ref_analysis': intensity_ref,
        'user_analysis': intensity_user,
        'cv_ref': cv_ref,
        'cv_user': cv_user
    }


def enhanced_intensity_analysis(gray_image, mask):
    """
    精密濃淡解析（Phase 1.5 機能強化）
    
    局所的濃度分析、勾配解析、筆圧推定精度向上を含む包括的な濃淡評価
    
    Args:
        gray_image: グレースケール画像（0-255）
        mask: 二値化マスク
        
    Returns:
        dict: 精密濃淡解析結果
    """
    if mask.sum() == 0:
        return _empty_intensity_result()
    
    # 基本解析
    basic_analysis = analyze_stroke_intensity(gray_image, mask)
    
    # 1. 局所的濃度分析
    local_analysis = _local_intensity_analysis(gray_image, mask)
    
    # 2. 濃度勾配解析
    gradient_analysis = _gradient_intensity_analysis(gray_image, mask)
    
    # 3. 筆圧推定精度向上
    pressure_analysis = _enhanced_pressure_estimation(gray_image, mask)
    
    # 統合評価
    enhanced_score = _calculate_enhanced_intensity_score(
        basic_analysis, local_analysis, gradient_analysis, pressure_analysis
    )
    
    return {
        **basic_analysis,
        'local_analysis': local_analysis,
        'gradient_analysis': gradient_analysis,
        'pressure_analysis': pressure_analysis,
        'enhanced_uniformity_score': enhanced_score,
        'analysis_type': 'enhanced'
    }

def _local_intensity_analysis(gray_image, mask):
    """局所的濃度分析（スライディングウィンドウ）"""
    import cv2
    
    stroke_pixels = 255 - gray_image[mask > 0]
    
    # スライディングウィンドウによる局所ムラ検出
    window_size = 15  # ウィンドウサイズ
    local_cvs = []
    
    # マスク領域での局所統計
    y_coords, x_coords = np.where(mask > 0)
    
    if len(y_coords) < window_size:
        return {
            'local_cv_mean': 0.0,
            'local_cv_std': 0.0,
            'local_uniformity': 1.0,
            'local_patches': 0
        }
    
    # サンプリング点の選定
    n_samples = min(50, len(y_coords) // 5)  # 適度なサンプリング数
    indices = np.linspace(0, len(y_coords) - 1, n_samples, dtype=int)
    
    for idx in indices:
        center_y, center_x = y_coords[idx], x_coords[idx]
        
        # ウィンドウ範囲設定
        y_min = max(0, center_y - window_size // 2)
        y_max = min(gray_image.shape[0], center_y + window_size // 2)
        x_min = max(0, center_x - window_size // 2)
        x_max = min(gray_image.shape[1], center_x + window_size // 2)
        
        # 局所領域での濃度解析
        local_mask = mask[y_min:y_max, x_min:x_max]
        local_gray = gray_image[y_min:y_max, x_min:x_max]
        
        if local_mask.sum() > 5:  # 十分な画素数がある場合
            local_stroke = 255 - local_gray[local_mask > 0]
            if len(local_stroke) > 3:
                local_mean = np.mean(local_stroke)
                local_std = np.std(local_stroke)
                local_cv = local_std / (local_mean + 1e-6)
                local_cvs.append(local_cv)
    
    if len(local_cvs) == 0:
        return {
            'local_cv_mean': 0.0,
            'local_cv_std': 0.0,
            'local_uniformity': 1.0,
            'local_patches': 0
        }
    
    local_cvs = np.array(local_cvs)
    local_cv_mean = np.mean(local_cvs)
    local_cv_std = np.std(local_cvs)
    
    # 局所均一性スコア（CV変動が小さいほど良い）
    local_uniformity = np.exp(-local_cv_std * 2.0) * np.exp(-local_cv_mean * 3.0)
    
    return {
        'local_cv_mean': float(local_cv_mean),
        'local_cv_std': float(local_cv_std),
        'local_uniformity': float(local_uniformity),
        'local_patches': len(local_cvs)
    }

def _gradient_intensity_analysis(gray_image, mask):
    """濃度勾配解析（エッジ強度評価）"""
    import cv2
    
    # Sobelフィルタによる勾配計算
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # マスク領域での勾配統計
    mask_gradient = gradient_magnitude[mask > 0]
    
    if len(mask_gradient) == 0:
        return {
            'edge_strength_mean': 0.0,
            'edge_strength_std': 0.0,
            'edge_sharpness': 0.0,
            'boundary_clarity': 0.0
        }
    
    edge_mean = np.mean(mask_gradient)
    edge_std = np.std(mask_gradient)
    
    # エッジ鮮明度（強い勾配の割合）
    strong_edge_threshold = np.percentile(mask_gradient, 75)
    strong_edge_ratio = np.sum(mask_gradient > strong_edge_threshold) / len(mask_gradient)
    
    # 境界明瞭度
    boundary_clarity = np.tanh(edge_mean / 50.0) * strong_edge_ratio
    
    return {
        'edge_strength_mean': float(edge_mean),
        'edge_strength_std': float(edge_std),
        'edge_sharpness': float(strong_edge_ratio),
        'boundary_clarity': float(boundary_clarity)
    }

def _enhanced_pressure_estimation(gray_image, mask):
    """筆圧推定精度向上（複数閾値、ヒストグラム解析）"""
    import cv2
    
    stroke_pixels = 255 - gray_image[mask > 0]
    
    if len(stroke_pixels) == 0:
        return {
            'pressure_consistency': 0.0,
            'histogram_analysis': {},
            'multi_threshold_analysis': {}
        }
    
    # ヒストグラム分布形状解析
    hist, bins = np.histogram(stroke_pixels, bins=32, range=(0, 255))
    hist_norm = hist / (np.sum(hist) + 1e-6)
    
    # 分布の特徴量
    hist_entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-10))
    hist_peak_count = len([i for i in range(1, len(hist)-1) 
                          if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist) * 0.1])
    
    # 複数閾値での二値化結果比較
    thresholds = [100, 128, 150, 180, 200]
    threshold_consistency = []
    
    for thresh in thresholds:
        strong_stroke_ratio = np.sum(stroke_pixels > thresh) / len(stroke_pixels)
        threshold_consistency.append(strong_stroke_ratio)
    
    # 閾値間の一貫性（段階的変化の滑らかさ）
    consistency_score = 1.0 - np.std(np.diff(threshold_consistency)) * 5.0
    consistency_score = np.clip(consistency_score, 0.0, 1.0)
    
    # 筆圧一貫性スコア
    pressure_consistency = (1.0 - hist_entropy / 5.5) * 0.5 + consistency_score * 0.5
    
    return {
        'pressure_consistency': float(pressure_consistency),
        'histogram_analysis': {
            'entropy': float(hist_entropy),
            'peak_count': int(hist_peak_count),
            'distribution_type': 'concentrated' if hist_entropy < 3.5 else 'dispersed'
        },
        'multi_threshold_analysis': {
            'threshold_ratios': [float(r) for r in threshold_consistency],
            'consistency_score': float(consistency_score)
        }
    }

def _calculate_enhanced_intensity_score(basic, local, gradient, pressure):
    """精密濃淡解析の統合スコア計算"""
    # 重み設定
    basic_weight = 0.4
    local_weight = 0.25
    gradient_weight = 0.2
    pressure_weight = 0.15
    
    # 各要素のスコア正規化
    basic_score = basic['uniformity_score']
    local_score = local['local_uniformity']
    gradient_score = gradient['boundary_clarity']
    pressure_score = pressure['pressure_consistency']
    
    # 統合スコア
    enhanced_score = (
        basic_weight * basic_score +
        local_weight * local_score +
        gradient_weight * gradient_score +
        pressure_weight * pressure_score
    )
    
    return float(enhanced_score)

def improved_width_analysis(mask):
    """
    改良線幅解析（Phase 1.5 機能強化）
    
    方向性を考慮した線幅測定、サンプリング密度最適化、ノイズ除去強化
    
    Args:
        mask: 二値化マスク
        
    Returns:
        dict: 改良線幅解析結果
    """
    if mask.sum() < 20:
        return _empty_width_result()
    
    # 基本線幅解析
    basic_cv = stroke_cv(mask)
    
    if basic_cv is None:
        return _empty_width_result()
    
    # 1. 方向性を考慮した線幅測定
    directional_analysis = _directional_width_analysis(mask)
    
    # 2. サンプリング密度の最適化
    sampling_analysis = _optimized_sampling_analysis(mask)
    
    # 3. ノイズ除去の強化
    noise_filtered_analysis = _enhanced_noise_filtering(mask)
    
    # 統合評価
    improved_cv = _calculate_improved_width_score(
        basic_cv, directional_analysis, sampling_analysis, noise_filtered_analysis
    )
    
    return {
        'basic_cv': basic_cv,
        'improved_cv': improved_cv,
        'directional_analysis': directional_analysis,
        'sampling_analysis': sampling_analysis,
        'noise_analysis': noise_filtered_analysis,
        'analysis_type': 'improved'
    }

def _directional_width_analysis(mask):
    """方向性を考慮した線幅測定"""
    import cv2
    from scipy import ndimage
    
    # 距離変換
    dist = _get_distance_transform_cached(mask)
    
    # 勾配計算（線の方向推定）
    grad_x = ndimage.sobel(mask.astype(float), axis=1)
    grad_y = ndimage.sobel(mask.astype(float), axis=0)
    
    # 線の方向角度計算
    angles = np.arctan2(grad_y, grad_x)
    
    # 方向別解析
    directions = {
        'horizontal': (np.abs(angles) < np.pi/8) | (np.abs(angles) > 7*np.pi/8),
        'vertical': (np.abs(angles - np.pi/2) < np.pi/8) | (np.abs(angles + np.pi/2) < np.pi/8),
        'diagonal_1': (np.abs(angles - np.pi/4) < np.pi/8) | (np.abs(angles + 3*np.pi/4) < np.pi/8),
        'diagonal_2': (np.abs(angles - 3*np.pi/4) < np.pi/8) | (np.abs(angles + np.pi/4) < np.pi/8)
    }
    
    directional_stats = {}
    
    for direction_name, direction_mask in directions.items():
        # 方向マスクと線マスクの積
        combined_mask = mask & direction_mask
        
        if combined_mask.sum() > 10:
            direction_widths = dist[combined_mask] * 2
            direction_widths = direction_widths[direction_widths > 0.5]
            
            if len(direction_widths) > 5:
                direction_cv = direction_widths.std() / (direction_widths.mean() + 1e-6)
                directional_stats[direction_name] = {
                    'cv': float(direction_cv),
                    'mean_width': float(direction_widths.mean()),
                    'pixel_count': int(len(direction_widths))
                }
            else:
                directional_stats[direction_name] = {'cv': 0.0, 'mean_width': 0.0, 'pixel_count': 0}
        else:
            directional_stats[direction_name] = {'cv': 0.0, 'mean_width': 0.0, 'pixel_count': 0}
    
    # 方向間の一貫性評価
    valid_cvs = [stats['cv'] for stats in directional_stats.values() if stats['pixel_count'] > 5]
    
    if len(valid_cvs) > 1:
        direction_consistency = 1.0 - np.std(valid_cvs) * 2.0
        direction_consistency = np.clip(direction_consistency, 0.0, 1.0)
    else:
        direction_consistency = 1.0 if len(valid_cvs) == 1 else 0.0
    
    return {
        'directional_stats': directional_stats,
        'direction_consistency': float(direction_consistency),
        'dominant_direction': max(directional_stats.keys(), 
                                key=lambda k: directional_stats[k]['pixel_count'])
    }

def _optimized_sampling_analysis(mask):
    """サンプリング密度の最適化"""
    import cv2
    from skimage import measure
    
    # 距離変換
    dist = _get_distance_transform_cached(mask)
    
    # スケルトン抽出による中心線の取得
    skeleton = _extract_skeleton(mask)
    
    if skeleton.sum() < 10:
        return {
            'uniform_sampling_cv': 0.0,
            'adaptive_sampling_cv': 0.0,
            'sampling_improvement': 0.0,
            'skeleton_length': 0
        }
    
    # 1. 等間隔サンプリング
    skeleton_points = np.column_stack(np.where(skeleton))
    
    if len(skeleton_points) > 20:
        # 等間隔でポイントを選択
        uniform_indices = np.linspace(0, len(skeleton_points)-1, 
                                    min(50, len(skeleton_points)//2), dtype=int)
        uniform_points = skeleton_points[uniform_indices]
        
        uniform_widths = []
        for point in uniform_points:
            y, x = point
            width = dist[y, x] * 2
            if width > 0.5:
                uniform_widths.append(width)
        
        uniform_cv = np.std(uniform_widths) / (np.mean(uniform_widths) + 1e-6) if len(uniform_widths) > 5 else 0.0
    else:
        uniform_cv = 0.0
    
    # 2. 適応的サンプリング（曲率に基づく）
    adaptive_cv = _adaptive_curvature_sampling(skeleton, dist)
    
    # サンプリング改善度
    basic_cv = stroke_cv(mask) or 0.0
    sampling_improvement = max(0.0, basic_cv - min(uniform_cv, adaptive_cv))
    
    return {
        'uniform_sampling_cv': float(uniform_cv),
        'adaptive_sampling_cv': float(adaptive_cv),
        'sampling_improvement': float(sampling_improvement),
        'skeleton_length': int(skeleton.sum())
    }

def _enhanced_noise_filtering(mask):
    """ノイズ除去の強化"""
    import cv2
    from scipy import ndimage
    
    # 距離変換
    dist = _get_distance_transform_cached(mask)
    
    # 1. 端点効果の除去改善
    # より厳密な端点検出
    endpoint_filtered_mask = _remove_endpoint_effects(mask, radius=3)
    
    if endpoint_filtered_mask.sum() > 0:
        endpoint_filtered_widths = dist[endpoint_filtered_mask] * 2
        endpoint_filtered_widths = endpoint_filtered_widths[endpoint_filtered_widths > 0.5]
        endpoint_filtered_cv = (endpoint_filtered_widths.std() / 
                              (endpoint_filtered_widths.mean() + 1e-6) 
                              if len(endpoint_filtered_widths) > 10 else 0.0)
    else:
        endpoint_filtered_cv = 0.0
    
    # 2. 小さなノイズの影響除去
    # モルフォロジー演算による精密なノイズ除去
    noise_filtered_mask = _remove_small_noise(mask)
    
    if noise_filtered_mask.sum() > 0:
        noise_filtered_widths = dist[noise_filtered_mask] * 2
        noise_filtered_widths = noise_filtered_widths[noise_filtered_widths > 0.5]
        noise_filtered_cv = (noise_filtered_widths.std() / 
                           (noise_filtered_widths.mean() + 1e-6) 
                           if len(noise_filtered_widths) > 10 else 0.0)
    else:
        noise_filtered_cv = 0.0
    
    # 3. 統計的外れ値除去
    outlier_filtered_cv = _statistical_outlier_removal(mask, dist)
    
    # 最も効果的なフィルタリング結果を選択
    basic_cv = stroke_cv(mask) or 1.0
    
    filtering_results = {
        'endpoint_filtered_cv': endpoint_filtered_cv,
        'noise_filtered_cv': noise_filtered_cv,
        'outlier_filtered_cv': outlier_filtered_cv
    }
    
    # 改善効果の評価
    improvements = {name: max(0.0, basic_cv - cv) 
                   for name, cv in filtering_results.items()}
    
    best_method = max(improvements.keys(), key=lambda k: improvements[k])
    
    return {
        **filtering_results,
        'best_filtering_method': best_method,
        'max_improvement': float(improvements[best_method]),
        'basic_cv': float(basic_cv)
    }

def _extract_skeleton(mask):
    """スケルトン抽出"""
    from skimage.morphology import skeletonize
    return skeletonize(mask > 0)

def _adaptive_curvature_sampling(skeleton, dist):
    """適応的曲率サンプリング"""
    # 曲率の高い部分で密にサンプリング
    if skeleton.sum() < 10:
        return 0.0
    
    # 簡易的な曲率推定（隣接点の角度変化）
    skeleton_points = np.column_stack(np.where(skeleton))
    
    if len(skeleton_points) < 10:
        return 0.0
    
    # 適応的サンプリング（詳細実装は簡略化）
    sample_indices = np.linspace(0, len(skeleton_points)-1, 
                               min(30, len(skeleton_points)//3), dtype=int)
    
    adaptive_widths = []
    for idx in sample_indices:
        y, x = skeleton_points[idx]
        width = dist[y, x] * 2
        if width > 0.5:
            adaptive_widths.append(width)
    
    return (np.std(adaptive_widths) / (np.mean(adaptive_widths) + 1e-6) 
            if len(adaptive_widths) > 5 else 0.0)

def _remove_endpoint_effects(mask, radius=3):
    """端点効果の除去"""
    import cv2
    
    # 端点検出（より精密に）
    kernel = np.ones((2*radius+1, 2*radius+1), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    
    return eroded > 0

def _remove_small_noise(mask):
    """小さなノイズの除去"""
    import cv2
    
    # 連結成分解析による小さなノイズ除去
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    
    # 最大の連結成分のみを保持
    if num_labels > 1:
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned_mask = (labels == largest_component).astype(np.uint8)
    else:
        cleaned_mask = mask
    
    return cleaned_mask

def _statistical_outlier_removal(mask, dist):
    """統計的外れ値除去"""
    widths = dist[mask > 0] * 2
    widths = widths[widths > 0.5]
    
    if len(widths) < 20:
        return 0.0
    
    # IQRによる外れ値除去
    q25, q75 = np.percentile(widths, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    
    filtered_widths = widths[(widths >= lower_bound) & (widths <= upper_bound)]
    
    return (filtered_widths.std() / (filtered_widths.mean() + 1e-6) 
            if len(filtered_widths) > 10 else 0.0)

def _calculate_improved_width_score(basic_cv, directional, sampling, noise):
    """改良線幅解析の統合スコア計算"""
    # 最も安定した結果を選択
    candidates = [
        basic_cv,
        sampling['uniform_sampling_cv'],
        sampling['adaptive_sampling_cv'],
        noise['endpoint_filtered_cv'],
        noise['noise_filtered_cv'],
        noise['outlier_filtered_cv']
    ]
    
    # 有効な（0でない）候補から最小値を選択
    valid_candidates = [cv for cv in candidates if cv > 0]
    
    if valid_candidates:
        improved_cv = min(valid_candidates)
    else:
        improved_cv = basic_cv
    
    # 方向一貫性による補正
    direction_factor = directional['direction_consistency']
    improved_cv *= (2.0 - direction_factor)  # 一貫性が高いほど改善
    
    return float(improved_cv)

def comprehensive_enhanced_black_score(gray_ref, mask_ref, gray_user, mask_user, 
                                     use_enhanced=True, use_improved_width=True):
    """
    包括的強化黒スコア（Phase 1.5 精密化機能統合）
    
    Args:
        gray_ref: お手本のグレースケール画像
        mask_ref: お手本の二値化マスク
        gray_user: ユーザーのグレースケール画像
        mask_user: ユーザーの二値化マスク
        use_enhanced: 精密濃淡解析を使用するか
        use_improved_width: 改良線幅解析を使用するか
        
    Returns:
        dict: 包括的強化黒スコア解析結果
    """
    # 基本評価（後方互換性）
    basic_result = comprehensive_black_score(gray_ref, mask_ref, gray_user, mask_user)
    
    if not use_enhanced and not use_improved_width:
        return basic_result
    
    # 精密化機能の適用
    enhanced_result = {**basic_result}
    
    # 1. 精密濃淡解析
    if use_enhanced:
        ref_enhanced_intensity = enhanced_intensity_analysis(gray_ref, mask_ref)
        user_enhanced_intensity = enhanced_intensity_analysis(gray_user, mask_user)
        
        # 精密濃淡類似度の計算
        enhanced_intensity_similarity = _calculate_enhanced_intensity_similarity(
            ref_enhanced_intensity, user_enhanced_intensity
        )
        
        enhanced_result.update({
            'ref_enhanced_intensity': ref_enhanced_intensity,
            'user_enhanced_intensity': user_enhanced_intensity,
            'enhanced_intensity_similarity': enhanced_intensity_similarity
        })
    
    # 2. 改良線幅解析
    if use_improved_width:
        ref_improved_width = improved_width_analysis(mask_ref)
        user_improved_width = improved_width_analysis(mask_user)
        
        # 改良線幅類似度の計算
        improved_width_similarity = _calculate_improved_width_similarity(
            ref_improved_width, user_improved_width
        )
        
        enhanced_result.update({
            'ref_improved_width': ref_improved_width,
            'user_improved_width': user_improved_width,
            'improved_width_similarity': improved_width_similarity
        })
    
    # 3. 統合スコアの再計算
    if use_enhanced or use_improved_width:
        enhanced_total_score = _calculate_enhanced_total_score(
            enhanced_result, use_enhanced, use_improved_width
        )
        enhanced_result['enhanced_total_score'] = enhanced_total_score
    
    enhanced_result['analysis_level'] = 'enhanced'
    
    return enhanced_result

def _calculate_enhanced_intensity_similarity(ref_analysis, user_analysis):
    """精密濃淡類似度の計算"""
    if (ref_analysis['enhanced_uniformity_score'] == 0.0 and 
        user_analysis['enhanced_uniformity_score'] == 0.0):
        return 1.0
    
    # 複数要素での類似度評価
    similarities = []
    
    # 1. 基本均一性
    basic_diff = abs(ref_analysis['enhanced_uniformity_score'] - 
                    user_analysis['enhanced_uniformity_score'])
    basic_sim = np.exp(-basic_diff * 3.0)
    similarities.append(('basic', basic_sim, 0.4))
    
    # 2. 局所一貫性
    local_diff = abs(ref_analysis['local_analysis']['local_uniformity'] - 
                    user_analysis['local_analysis']['local_uniformity'])
    local_sim = np.exp(-local_diff * 2.5)
    similarities.append(('local', local_sim, 0.25))
    
    # 3. 境界明瞭度
    boundary_diff = abs(ref_analysis['gradient_analysis']['boundary_clarity'] - 
                       user_analysis['gradient_analysis']['boundary_clarity'])
    boundary_sim = np.exp(-boundary_diff * 2.0)
    similarities.append(('boundary', boundary_sim, 0.2))
    
    # 4. 筆圧一貫性
    pressure_diff = abs(ref_analysis['pressure_analysis']['pressure_consistency'] - 
                       user_analysis['pressure_analysis']['pressure_consistency'])
    pressure_sim = np.exp(-pressure_diff * 2.0)
    similarities.append(('pressure', pressure_sim, 0.15))
    
    # 重み付き統合
    total_similarity = sum(sim * weight for _, sim, weight in similarities)
    
    return float(total_similarity)

def _calculate_improved_width_similarity(ref_analysis, user_analysis):
    """改良線幅類似度の計算"""
    ref_cv = ref_analysis.get('improved_cv')
    user_cv = user_analysis.get('improved_cv')
    
    if ref_cv is None or user_cv is None:
        return 0.0
    
    # 基本CV類似度
    sigma = ref_cv * 0.5 + 1e-6
    basic_sim = np.exp(-((user_cv - ref_cv) ** 2) / (2 * sigma ** 2))
    
    # 方向一貫性の類似度
    ref_consistency = ref_analysis['directional_analysis']['direction_consistency']
    user_consistency = user_analysis['directional_analysis']['direction_consistency']
    consistency_sim = 1.0 - abs(ref_consistency - user_consistency)
    
    # サンプリング改善度の考慮
    ref_improvement = ref_analysis['sampling_analysis']['sampling_improvement']
    user_improvement = user_analysis['sampling_analysis']['sampling_improvement']
    improvement_factor = 1.0 + min(ref_improvement, user_improvement) * 0.2
    
    # 統合類似度
    total_similarity = (basic_sim * 0.7 + consistency_sim * 0.3) * improvement_factor
    
    return float(np.clip(total_similarity, 0.0, 1.0))

def _calculate_enhanced_total_score(enhanced_result, use_enhanced, use_improved_width):
    """精密化機能を含む総合スコア計算"""
    # 基本スコア
    base_total = enhanced_result['total_score']
    
    # 精密化要素の重み
    enhanced_weight = 0.15 if use_enhanced else 0.0
    improved_width_weight = 0.10 if use_improved_width else 0.0
    base_weight = 1.0 - enhanced_weight - improved_width_weight
    
    enhanced_total = base_weight * base_total
    
    # 精密濃淡解析スコア
    if use_enhanced:
        enhanced_intensity_score = enhanced_result.get('enhanced_intensity_similarity', 0.0)
        enhanced_total += enhanced_weight * enhanced_intensity_score
    
    # 改良線幅解析スコア
    if use_improved_width:
        improved_width_score = enhanced_result.get('improved_width_similarity', 0.0)
        enhanced_total += improved_width_weight * improved_width_score
    
    return float(enhanced_total)

def _empty_intensity_result():
    """空マスク用の結果"""
    return {
        'mean_intensity': 0.0,
        'intensity_std': 0.0,
        'intensity_cv': 0.0,
        'uniformity_score': 0.0,
        'local_analysis': {
            'local_cv_mean': 0.0,
            'local_cv_std': 0.0,
            'local_uniformity': 0.0,
            'local_patches': 0
        },
        'gradient_analysis': {
            'edge_strength_mean': 0.0,
            'edge_strength_std': 0.0,
            'edge_sharpness': 0.0,
            'boundary_clarity': 0.0
        },
        'pressure_analysis': {
            'pressure_consistency': 0.0,
            'histogram_analysis': {},
            'multi_threshold_analysis': {}
        },
        'enhanced_uniformity_score': 0.0,
        'analysis_type': 'enhanced'
    }

def _empty_width_result():
    """空マスク用の結果"""
    return {
        'basic_cv': None,
        'improved_cv': None,
        'directional_analysis': {
            'directional_stats': {},
            'direction_consistency': 0.0,
            'dominant_direction': 'none'
        },
        'sampling_analysis': {
            'uniform_sampling_cv': 0.0,
            'adaptive_sampling_cv': 0.0,
            'sampling_improvement': 0.0,
            'skeleton_length': 0
        },
        'noise_analysis': {
            'endpoint_filtered_cv': 0.0,
            'noise_filtered_cv': 0.0,
            'outlier_filtered_cv': 0.0,
            'best_filtering_method': 'none',
            'max_improvement': 0.0,
            'basic_cv': 0.0
        },
        'analysis_type': 'improved'
    }
