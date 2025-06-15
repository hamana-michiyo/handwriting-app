# 検証プログラム

このフォルダには、手書き文字評価システムの性能を検証するためのテストプログラムが含まれています。

## プログラム一覧

### 1. shape_evaluation_comparison.py
**形状評価手法比較テスト**

異なる形状評価手法（元IoU、位置補正IoU、Hu類似度、ハイブリッド評価）の性能を比較検証します。

```bash
python validation/shape_evaluation_comparison.py
```

**検証内容:**
- 同じ形状での位置ずれ影響
- 異なる形状での識別性能
- 重み設定の感度分析

### 2. test_position_robustness.py
**位置ロバスト性テスト**

位置ずれに対する形状評価の頑健性を検証します。

```bash
python validation/test_position_robustness.py
```

**検証内容:**
- 同一形状・異なる位置での評価安定性
- 位置補正機能の有効性
- テンプレートマッチング精度

### 3. test_scale_robustness.py
**スケールロバスト性テスト**

サイズ違いの相似形に対する評価性能を検証します。

```bash
python validation/test_scale_robustness.py
```

**検証内容:**
- 異なるサイズの同形状評価
- マルチスケール探索機能
- 相似形識別精度

### 4. test_intensity_analysis.py
**濃淡解析機能テスト**

線の濃淡均一性解析と包括的黒スコア評価を検証します。

```bash
python validation/test_intensity_analysis.py
```

**検証内容:**
- 濃度均一性解析の動作確認
- 薄すぎ・ムラ検出機能
- 線幅安定性との統合評価

## 期待される結果

### 位置ロバスト性
- 同一形状（位置ずれ）: スコア = 1.000（完璧）
- 位置補正による大幅改善を確認

### スケールロバスト性  
- 小さい円(2/3サイズ): スコア ≈ 0.896
- 大きい円(4/3サイズ): スコア ≈ 0.914
- 2倍円: スコア ≈ 0.995

### 形状識別性
- 円 vs 正方形: スコア ≈ 0.867（適切な区別）

### 線質評価性能
- 均一な濃度: 均一性スコア ≈ 0.8+
- 濃淡ムラあり: 均一性スコア大幅低下
- 統合評価: 線幅(60%) + 濃淡(40%)で総合判定

## 実行環境

Python 3.11+ + OpenCV + NumPy が必要です。

```bash
# 依存関係のインストール
pip install -r requirements.txt

# 全検証プログラムの実行
cd /workspace
python validation/shape_evaluation_comparison.py
python validation/test_position_robustness.py  
python validation/test_scale_robustness.py
python validation/test_intensity_analysis.py
```

## 注意事項

- これらのプログラムは検証・テスト目的のため、実際の評価パイプラインには含まれません
- 新しい機能を追加した際は、対応する検証プログラムの更新も推奨されます
- 性能回帰を検出するため、定期的な実行が推奨されます
