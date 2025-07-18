# 📝 美文字アプリ向け 機械学習データ管理ツール概要

## 🎯 目的
手書き文字の評価データを効率的に収集・管理し、  
今後の機械学習（例：CNNでの評価予測モデル）に活用できる形に整えるためのツール。

---

## 🔧 主な機能構成

### 1. 📤 データ作成・画像読み込み

- A4紙に3〜5字書いたものを **スキャナ or カメラ画像として読み込み**
- フォーマット：
  - **文字は正方形マス＋十字線の枠内に書かれている**
  - 読込対象範囲の四隅に **トンボ（基準点）**あり

- 記入者番号を入力する

- 処理内容：
  - OpenCVで **トンボ検出・歪み補正**
  - 補正済みの読込範囲の **プレビュー表示**
  - プレビューでOKなら補正済みの読込範囲の画像をバックエンドに送信、NGなら再度カメラで撮り直し
  - バックエンド(python)の処理:
    - 自動でマス切り出し、"記入者番号_文字_日付.jpg"として保存
    - DBに画像情報を保存する


---

### 2. 🖼️ 画像一覧・管理画面

- 切り出された文字画像を **グリッド表示**
- 各マスに「文字名（ラベル）」と「記入者番号」が表示
- 評価状態（未評価・済）をアイコン等で表示

---

### 3. ✏️ 評価入力画面

- 表示:
  - 記入者番号
  - 文字の画像
  - 評価用の入力:
    - 1文字ずつ、以下の4観点を評価：
      - **形**（文字の形が整っているか）
      - **黒**（線の強さ・安定）
      - **白**（余白の美しさ）
      - **場**（紙面での位置バランス）
    - 4観点の評価はそれぞれ**1～10の数値＋コメント**がある

- 前後の画像へ移動ボタン

---

### 4. 📁 保存・エクスポート(参考)

- 評価データの保存形式：
  - JSON形式で保存（後からDB化）
- 画像と評価データを **文字単位で紐付け保存**
- 評価データから **再学習セット作成が可能**

```
{
  "image_id": "S001_春_20250615",
  "shape": 8,
  "shape_comment": "バランス良好",
  "black": 7,
  "black_comment": "線の強弱がやや不安定",
  "white": 9,
  "white_comment": "",
  "placement": 10,
  "placement_comment": "中央に整っている",
  "evaluator": "Ishikawa",
  "evaluated_at": "2025-06-15T09:30:00"
}
````

---

## 🛠️ 技術スタック候補

- **UIアプリ**：Flutter（モバイル・タブレットにも対応）
- **画像処理**：Python（OpenCV＋NumPy＋Pillow）
- **バックエンド**：FastAPI, PostgreSQL or Firebase
- **ストレージ**：ローカル or Google Drive連携など

---

## 🔮 将来的な拡張アイデア

- 記入者番号、評価の数字やコメントもOCRで取り込む
- AIによる **自動スコア予測＋人間との比較表示**
- 評価の傾向可視化（先生ごとの傾向など）
- Web版＋クラウド同期でチーム対応

---


## 画面UI案


### 1. TOP画面
 - 新規データ作成ボタン→画像読み込み画面へ
 - 保存データ一覧ボタン→データ一覧画面へ

### 2. 画像読み込み画面

 - 📸 画像選択（スキャナ or 画像ファイル）
 - ✅ トンボ検出・歪み補正（OpenCVに連携）
 - 🧩 自動でマス目を切り出してリスト表示
 - マス番号 or 文字ラベル付き
 - プレビュー一覧表示（Grid）

### 3.データ一覧画面

 - 検索条件
	- 文字で絞り込み
	- 未評価/評価済み
	- 評価者

 - 選択したら評価入力画面へ

### 4.評価入力画面

 - ✏️ 評価対象の文字画像（1つ表示 or スクロール可）
 - 🧮 評価項目ごとの入力欄（数値 or セレクト）
 - 形（スライダー or 1〜10）＋コメント欄
 - 黒（スライダー）＋コメント欄
 - 白（スライダー）＋コメント欄
 - 場（スライダー）＋コメント欄
 - 🔄 前後の画像に移動ボタン
 - ✅ 入力済みマーク
 - 評価者



