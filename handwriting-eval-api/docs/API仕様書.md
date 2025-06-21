# 美文字アプリ 統合API仕様書 2025-06-21

## 概要

既存の手書き文字評価APIに画像管理・OCR機能を拡張し、Flutterアプリからの画像処理とデータ管理を行うAPIシステム。

## システム構成

```
Flutter App → FastAPI Server → 画像処理・OCR → データベース・ファイル管理
```

## API エンドポイント設計

### 1. 画像アップロード・処理API

#### `POST /api/images/upload`
撮影した記入用紙画像をアップロードし、自動処理を実行

**リクエスト**
```json
{
  "image": "base64_encoded_image_data",
  "writer_id": "記入者番号（オプション）",
  "session_id": "セッションID（オプション）",
  "metadata": {
    "timestamp": "2025-06-21T10:30:00Z",
    "device_info": "iPhone 14 Pro"
  }
}
```

**レスポンス**
```json
{
  "success": true,
  "upload_id": "upload_20250621_103045_001",
  "message": "画像処理が完了しました",
  "processing_result": {
    "detected_tombo": true,
    "correction_applied": true,
    "extracted_characters": 3,
    "ocr_results": {
      "writer_number": "S001",
      "characters": ["春", "夏", "秋"],
      "evaluations": {
        "spring_shape": 8,
        "spring_black": 7,
        "spring_white": 9,
        "spring_placement": 8,
        "summer_shape": 7,
        "summer_black": 8,
        "summer_white": 8,
        "summer_placement": 9
      },
      "comments": {
        "character_1": {
          "shape_comment": {
            "text": "バランスが良い",
            "confidence": 0.92
          },
          "black_comment": {
            "text": "線の強弱不安定",
            "confidence": 0.88
          },
          "white_comment": {
            "text": "余白美しい",
            "confidence": 0.95
          },
          "placement_comment": {
            "text": "中央に整っている",
            "confidence": 0.90
          }
        },
        "character_2": {
          "shape_comment": {
            "text": "はらいが弱い",
            "confidence": 0.85
          },
          "black_comment": {
            "text": "濃淡ムラあり",
            "confidence": 0.87
          }
        }
      }
    }
  },
  "character_images": [
    {
      "character_id": "S001_春_20250621",
      "character": "春",
      "image_path": "/api/images/S001_春_20250621.jpg",
      "thumbnail_path": "/api/images/thumbnails/S001_春_20250621_thumb.jpg",
      "bbox": {"x": 100, "y": 150, "width": 200, "height": 200},
      "evaluation_comments": {
        "shape": "バランスが良い",
        "black": "線の強弱不安定", 
        "white": "余白美しい",
        "placement": "中央に整っている"
      }
    }
  ],
  "evaluation_data_path": "/api/data/S001_20250621_evaluation.json"
}
```

### 2. 画像取得API

#### `GET /api/images/{image_id}`
個別の文字画像を取得

**パラメータ**
- `image_id`: 文字画像ID (例: S001_春_20250621)
- `type`: original | thumbnail | corrected (デフォルト: original)

**レスポンス**
- Content-Type: image/jpeg
- 画像バイナリデータ

#### `GET /api/images/list`
文字画像一覧を取得

**クエリパラメータ**
- `writer_id`: 記入者番号でフィルタ
- `character`: 文字でフィルタ
- `date_from`, `date_to`: 日付範囲
- `evaluated`: true | false | all (評価済みかどうか)
- `limit`: 取得件数制限
- `offset`: オフセット

**レスポンス**
```json
{
  "total": 156,
  "images": [
    {
      "character_id": "S001_春_20250621",
      "writer_id": "S001",
      "character": "春",
      "created_at": "2025-06-21T10:30:00Z",
      "evaluated": true,
      "evaluation_scores": {
        "shape": 8,
        "black": 7,
        "white": 9,
        "placement": 8
      },
      "image_url": "/api/images/S001_春_20250621",
      "thumbnail_url": "/api/images/S001_春_20250621?type=thumbnail"
    }
  ]
}
```

### 3. 評価データAPI

#### `GET /api/evaluations/{character_id}`
特定文字の評価データを取得

**レスポンス**
```json
{
  "character_id": "S001_春_20250621",
  "writer_id": "S001",
  "character": "春",
  "created_at": "2025-06-21T10:30:00Z",
  "manual_evaluation": {
    "shape": 8,
    "shape_comment": "バランス良好",
    "black": 7,
    "black_comment": "線の強弱がやや不安定",
    "white": 9,
    "white_comment": "",
    "placement": 8,
    "placement_comment": "中央に整っている",
    "evaluator": "Teacher01",
    "evaluated_at": "2025-06-21T11:00:00Z"
  },
  "ocr_evaluation": {
    "shape": 8,
    "black": 7,
    "white": 9,
    "placement": 8,
    "confidence": 0.85
  },
  "ai_evaluation": {
    "total_score": 90.2,
    "shape": 89.5,
    "black": 92.3,
    "white": 85.7,
    "placement": 94.1,
    "analysis": "高精度評価結果"
  }
}
```

#### `PUT /api/evaluations/{character_id}`
評価データを更新

**リクエスト**
```json
{
  "shape": 8,
  "shape_comment": "バランス良好",
  "black": 7,
  "black_comment": "線の強弱がやや不安定",
  "white": 9,
  "white_comment": "",
  "placement": 8,
  "placement_comment": "中央に整っている",
  "evaluator": "Teacher01"
}
```

### 4. 統計・管理API

#### `GET /api/statistics`
データ統計情報を取得

**レスポンス**
```json
{
  "total_images": 156,
  "total_writers": 24,
  "evaluated_images": 89,
  "unevaluated_images": 67,
  "character_distribution": {
    "春": 24,
    "夏": 24,
    "秋": 24,
    "冬": 24
  },
  "recent_uploads": 12,
  "average_scores": {
    "shape": 7.8,
    "black": 8.2,
    "white": 7.5,
    "placement": 8.9
  }
}
```

## 画像処理パイプライン

### 1. トンボ検出・歪み補正
```python
def process_uploaded_image(image_data):
    # 1. Base64デコード
    image = decode_base64_image(image_data)
    
    # 2. トンボ検出
    tombo_points = detect_tombo_marks(image)
    
    # 3. 歪み補正
    corrected_image = apply_perspective_correction(image, tombo_points)
    
    # 4. 文字領域切り出し
    character_regions = extract_character_regions(corrected_image)
    
    return corrected_image, character_regions
```

### 2. OCR処理
```python
def perform_ocr_analysis(image, character_regions):
    results = {
        "writer_number": None,
        "characters": [],
        "evaluations": {},
        "comments": {}
    }
    
    # 記入者番号OCR
    writer_region = extract_writer_number_region(image)
    results["writer_number"] = ocr_writer_number(writer_region)
    
    # 文字OCR
    for i, region in enumerate(character_regions):
        char = ocr_character(region)
        results["characters"].append(char)
    
    # 評価欄OCR（右端の数字）
    eval_region = extract_evaluation_region(image)
    results["evaluations"] = ocr_evaluations(eval_region)
    
    # コメント欄OCR（評価数字の左側）
    comment_region = extract_comment_region(image)
    results["comments"] = ocr_comments(comment_region)
    
    return results

def extract_comment_region(image):
    """評価コメント欄の座標を特定して切り出し"""
    # 記入用紙の右側領域から、数字欄の左側の横長エリアを抽出
    height, width = image.shape[:2]
    
    # 想定座標（記入用紙レイアウトに基づく）
    comment_regions = {
        "character_1": {
            "shape_comment": (width*0.6, height*0.3, width*0.85, height*0.35),
            "black_comment": (width*0.6, height*0.35, width*0.85, height*0.4),
            "white_comment": (width*0.6, height*0.4, width*0.85, height*0.45),
            "placement_comment": (width*0.6, height*0.45, width*0.85, height*0.5)
        },
        "character_2": {
            "shape_comment": (width*0.6, height*0.55, width*0.85, height*0.6),
            "black_comment": (width*0.6, height*0.6, width*0.85, height*0.65),
            "white_comment": (width*0.6, height*0.65, width*0.85, height*0.7),
            "placement_comment": (width*0.6, height*0.7, width*0.85, height*0.75)
        },
        "character_3": {
            "shape_comment": (width*0.6, height*0.8, width*0.85, height*0.85),
            "black_comment": (width*0.6, height*0.85, width*0.85, height*0.9),
            "white_comment": (width*0.6, height*0.9, width*0.85, height*0.95),
            "placement_comment": (width*0.6, height*0.95, width*0.85, height*1.0)
        }
    }
    
    return comment_regions

def ocr_comments(comment_regions):
    """コメント欄のテキストをOCR処理"""
    import pytesseract
    
    results = {}
    
    for char_key, char_comments in comment_regions.items():
        results[char_key] = {}
        
        for eval_type, coords in char_comments.items():
            x1, y1, x2, y2 = map(int, coords)
            comment_image = image[y1:y2, x1:x2]
            
            # 日本語OCR設定
            custom_config = r'--oem 3 --psm 6 -l jpn'
            
            # OCR実行
            text = pytesseract.image_to_string(
                comment_image, 
                config=custom_config
            ).strip()
            
            # ノイズ除去・クリーニング
            cleaned_text = clean_comment_text(text)
            
            results[char_key][eval_type] = {
                "text": cleaned_text,
                "confidence": get_ocr_confidence(comment_image, text),
                "bbox": coords
            }
    
    return results

def clean_comment_text(text):
    """OCRで取得したコメントテキストのクリーニング"""
    import re
    
    # 不要な記号・改行を除去
    text = re.sub(r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', '', text)
    
    # 余分な空白を除去
    text = ' '.join(text.split())
    
    # 一般的なOCR誤認識を修正
    corrections = {
        '0': 'O',  # 数字の0 → アルファベットのO
        '1': 'I',  # 数字の1 → アルファベットのI
        # その他、よくある誤認識パターンを追加
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    return text
```

### 3. ファイル保存
```python
def save_processed_images(upload_id, corrected_image, character_regions, ocr_results):
    base_path = f"data/images/{upload_id}"
    
    # 元画像保存
    save_image(corrected_image, f"{base_path}/original.jpg")
    
    # 個別文字画像保存
    character_paths = []
    for i, (region, char) in enumerate(zip(character_regions, ocr_results["characters"])):
        filename = f"{ocr_results['writer_number']}_{char}_{datetime.now().strftime('%Y%m%d')}.jpg"
        path = f"{base_path}/{filename}"
        save_image(region, path)
        
        # サムネイル作成
        thumbnail = create_thumbnail(region, size=(150, 150))
        thumb_path = f"{base_path}/thumbnails/{filename}"
        save_image(thumbnail, thumb_path)
        
        character_paths.append({
            "character": char,
            "path": path,
            "thumbnail": thumb_path
        })
    
    return character_paths
```

## データベース設計

### テーブル構成

#### uploads テーブル
```sql
CREATE TABLE uploads (
    id VARCHAR(50) PRIMARY KEY,
    original_image_path VARCHAR(255),
    processed_image_path VARCHAR(255),
    writer_id VARCHAR(20),
    upload_timestamp TIMESTAMP,
    processing_status VARCHAR(20),
    metadata JSON
);
```

#### character_images テーブル
```sql
CREATE TABLE character_images (
    character_id VARCHAR(50) PRIMARY KEY,
    upload_id VARCHAR(50),
    writer_id VARCHAR(20),
    character VARCHAR(10),
    image_path VARCHAR(255),
    thumbnail_path VARCHAR(255),
    bbox JSON,
    created_at TIMESTAMP,
    FOREIGN KEY (upload_id) REFERENCES uploads(id)
);
```

#### evaluations テーブル
```sql
CREATE TABLE evaluations (
    id SERIAL PRIMARY KEY,
    character_id VARCHAR(50),
    evaluation_type VARCHAR(20), -- 'manual', 'ocr', 'ai'
    shape_score INTEGER,
    shape_comment TEXT,
    black_score INTEGER,
    black_comment TEXT,
    white_score INTEGER,
    white_comment TEXT,
    placement_score INTEGER,
    placement_comment TEXT,
    evaluator VARCHAR(50),
    evaluated_at TIMESTAMP,
    FOREIGN KEY (character_id) REFERENCES character_images(character_id)
);
```

## 実装技術スタック

### バックエンド
- **FastAPI**: RESTful API
- **OpenCV**: 画像処理・トンボ検出・歪み補正
- **Pillow**: 画像操作
- **Tesseract/PaddleOCR**: OCR処理
- **SQLite/PostgreSQL**: データベース
- **SQLAlchemy**: ORM

### ディレクトリ構造
```
handwriting-eval-api/
├── api_server.py              # FastAPI メインサーバー
├── src/
│   ├── api/                   # API エンドポイント
│   │   ├── images.py          # 画像関連API
│   │   ├── evaluations.py     # 評価関連API
│   │   └── statistics.py      # 統計API
│   ├── core/                  # コア機能
│   │   ├── image_processor.py # 画像処理パイプライン
│   │   ├── ocr_engine.py     # OCR処理
│   │   └── file_manager.py   # ファイル管理
│   ├── models/               # データモデル
│   │   ├── database.py       # DB設定
│   │   └── schemas.py        # Pydanticスキーマ
│   └── eval/                 # 既存の評価システム
├── data/
│   ├── images/               # アップロード画像
│   ├── processed/            # 処理済み画像
│   └── evaluations/          # 評価JSONファイル
└── tests/
```

## セキュリティ・運用考慮事項

### セキュリティ
- API認証（JWT トークン）
- ファイルアップロード制限（サイズ・形式）
- SQLインジェクション対策
- CORS設定

### 運用
- ログ記録（アップロード・処理履歴）
- エラーハンドリング
- 画像ストレージ容量管理
- バックアップ機能

## 機械学習におけるコメントデータの活用

### 1. 自然言語処理による感情分析
```python
# コメントの感情極性分析
def analyze_comment_sentiment(comment_text):
    from transformers import pipeline
    
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="cl-tohoku/bert-base-japanese-whole-word-masking"
    )
    
    result = sentiment_analyzer(comment_text)
    return {
        "sentiment": result[0]["label"],  # POSITIVE/NEGATIVE
        "confidence": result[0]["score"],
        "keywords": extract_keywords(comment_text)
    }

# コメントカテゴリ分類
comment_categories = {
    "技術的指摘": ["線", "はらい", "とめ", "バランス", "太さ"],
    "美的評価": ["美しい", "整っている", "きれい", "汚い"],
    "構造的問題": ["傾いている", "大きい", "小さい", "位置"],
    "総合評価": ["良い", "悪い", "普通", "優秀"]
}
```

### 2. 評価予測モデルの精度向上
```python
# マルチモーダル学習（画像 + テキスト）
class MultimodalEvaluator(nn.Module):
    def __init__(self):
        # 画像特徴抽出
        self.image_encoder = ResNet50(pretrained=True)
        
        # テキスト特徴抽出
        self.text_encoder = BertJapanese.from_pretrained(
            'cl-tohoku/bert-base-japanese'
        )
        
        # 特徴融合
        self.fusion_layer = nn.Linear(2048 + 768, 512)
        self.score_predictor = nn.Linear(512, 4)  # 4軸評価
    
    def forward(self, image, comment_text):
        # 画像特徴
        img_features = self.image_encoder(image)
        
        # テキスト特徴
        text_features = self.text_encoder(comment_text).pooler_output
        
        # 特徴結合
        combined = torch.cat([img_features, text_features], dim=1)
        fused_features = self.fusion_layer(combined)
        
        # スコア予測
        scores = self.score_predictor(fused_features)
        return scores
```

### 3. データセット構造の拡張
```json
{
  "dataset_version": "2.0",
  "samples": [
    {
      "image_id": "S001_春_20250621",
      "image_path": "images/S001_春_20250621.jpg",
      "character": "春",
      "scores": {
        "shape": 8,
        "black": 7,
        "white": 9,
        "placement": 8
      },
      "comments": {
        "shape": {
          "text": "バランスが良い",
          "sentiment": "POSITIVE",
          "keywords": ["バランス", "良い"],
          "category": "美的評価"
        },
        "black": {
          "text": "線の強弱不安定",
          "sentiment": "NEGATIVE", 
          "keywords": ["線", "強弱", "不安定"],
          "category": "技術的指摘"
        }
      },
      "evaluator_info": {
        "id": "teacher_001",
        "experience_years": 15,
        "specialty": "楷書"
      }
    }
  ]
}
```

### データセットエクスポートAPI

#### `GET /api/dataset/export`
機械学習用データセットを生成・エクスポート

**クエリパラメータ**
- `format`: csv | json | tfrecord | pytorch
- `split_ratio`: "0.7,0.2,0.1" (train,val,test)
- `augmentation`: true | false
- `normalize_scores`: true | false

**レスポンス**
```json
{
  "dataset_id": "dataset_20250621_001",
  "download_url": "/api/dataset/download/dataset_20250621_001.zip",
  "metadata": {
    "total_samples": 500,
    "train_samples": 350,
    "val_samples": 100,
    "test_samples": 50,
    "character_distribution": {
      "春": 125, "夏": 125, "秋": 125, "冬": 125
    },
    "score_statistics": {
      "shape": {"mean": 7.8, "std": 1.2},
      "black": {"mean": 8.1, "std": 1.0}
    }
  }
}
```

### PyTorch Dataset形式
```python
class BimojiDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, annotations_file, transform=None):
        self.data = pd.read_json(annotations_file)
        self.transform = transform
    
    def __getitem__(self, idx):
        image = Image.open(self.data.iloc[idx]['image_path'])
        scores = {
            'shape': self.data.iloc[idx]['shape_score'],
            'black': self.data.iloc[idx]['black_score'],
            'white': self.data.iloc[idx]['white_score'],
            'placement': self.data.iloc[idx]['placement_score']
        }
        
        if self.transform:
            image = self.transform(image)
            
        return image, scores
```

### TensorFlow Dataset形式
```python
def create_tf_dataset(data_dir):
    def parse_function(image_path, scores):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        
        return image, scores
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, scores))
    dataset = dataset.map(parse_function)
    return dataset
```

### データ拡張機能
```python
def augment_dataset(image_dir, output_dir, multiplier=3):
    augmentations = [
        {"rotation": (-5, 5)},
        {"brightness": (0.8, 1.2)},
        {"contrast": (0.9, 1.1)},
        {"noise": {"type": "gaussian", "std": 0.01}},
        {"elastic_transform": {"alpha": 1, "sigma": 50}}
    ]
    
    # 各画像に対して拡張データを生成
    # スコアは元データと同じ値を使用
```

## Flutter側OCR確認・訂正システム

### 5. OCR結果確認・訂正API

#### `GET /api/ocr/review/{upload_id}`
OCR結果を確認・訂正用データ形式で取得

**レスポンス**
```json
{
  "upload_id": "upload_20250621_103045_001",
  "original_image_url": "/api/images/upload_20250621_103045_001/original.jpg",
  "review_items": [
    {
      "field_type": "writer_number",
      "field_id": "writer_number",
      "label": "記入者番号",
      "ocr_text": "S001",
      "confidence": 0.95,
      "needs_review": false,
      "bbox": {"x": 100, "y": 50, "width": 80, "height": 30}
    },
    {
      "field_type": "character",
      "field_id": "character_1",
      "label": "文字1",
      "ocr_text": "春",
      "confidence": 0.88,
      "needs_review": true,
      "bbox": {"x": 150, "y": 200, "width": 200, "height": 200}
    },
    {
      "field_type": "comment",
      "field_id": "comment_shape_1",
      "label": "形コメント1",
      "ocr_text": "バランスが艮い",
      "confidence": 0.65,
      "needs_review": true,
      "suggested_corrections": ["バランスが良い", "バランスが悪い"],
      "bbox": {"x": 400, "y": 300, "width": 150, "height": 25}
    }
  ],
  "total_items": 12,
  "needs_review_count": 5
}
```

#### `POST /api/ocr/submit-corrections`
訂正されたOCR結果を送信

**リクエスト**
```json
{
  "upload_id": "upload_20250621_103045_001",
  "corrections": [
    {
      "field_id": "comment_shape_1",
      "original_text": "バランスが艮い",
      "corrected_text": "バランスが良い",
      "original_confidence": 0.65,
      "correction_method": "user_manual"
    },
    {
      "field_id": "character_2",
      "original_text": "夏",
      "corrected_text": "夏",
      "original_confidence": 0.82,
      "correction_method": "user_confirmed"
    }
  ],
  "review_time_seconds": 45,
  "user_feedback": {
    "overall_satisfaction": 4,
    "ocr_accuracy_rating": 3,
    "ui_usability_rating": 5
  }
}
```

**レスポンス**
```json
{
  "success": true,
  "processed_corrections": 2,
  "final_data": {
    "writer_id": "S001",
    "characters": ["春", "夏", "秋"],
    "evaluations": {
      "spring_shape": 8,
      "spring_shape_comment": "バランスが良い"
    }
  },
  "feedback_recorded": true,
  "triggers_retraining": false
}
```

### 6. OCR学習・改善API

#### `POST /api/ocr/feedback`
OCR修正データを収集してモデル改善に活用

**リクエスト**
```json
{
  "image_id": "upload_20250621_103045_001",
  "corrections": [
    {
      "field": "comment_shape_1",
      "field_type": "comment",
      "original_ocr": "バランスが艮い",
      "corrected_text": "バランスが良い",
      "confidence": 0.65,
      "image_region": {
        "x": 400, "y": 300, "width": 150, "height": 25
      }
    }
  ],
  "user_time_spent": 45,
  "correction_difficulty": "easy", // easy, medium, hard
  "user_expertise": "teacher" // student, teacher, expert
}
```

#### `GET /api/ocr/training-stats`
OCR学習データの統計情報

**レスポンス**
```json
{
  "total_corrections": 1250,
  "corrections_by_field": {
    "writer_number": 45,
    "characters": 234,
    "comments": 971
  },
  "accuracy_improvements": {
    "writer_number": {
      "before": 0.85,
      "after": 0.94,
      "improvement": 0.09
    },
    "comments": {
      "before": 0.62,
      "after": 0.78,
      "improvement": 0.16
    }
  },
  "next_retraining_threshold": 1500,
  "last_model_update": "2025-06-15T10:30:00Z"
}
```

#### `POST /api/ocr/retrain`
蓄積された修正データでOCRモデルを再学習

**リクエスト**
```json
{
  "training_mode": "incremental", // full, incremental
  "field_types": ["comments", "characters"], // 特定フィールドのみ学習
  "min_correction_count": 100,
  "validation_split": 0.2
}
```

**レスポンス**
```json
{
  "training_job_id": "training_20250621_001",
  "estimated_duration_minutes": 45,
  "status": "started",
  "training_data_size": 1250,
  "model_version": "v2.1",
  "progress_url": "/api/ocr/training-progress/training_20250621_001"
}
```

## Flutter UIアーキテクチャ設計

### OCR確認・訂正画面の実装

#### 1. 画面構成
```dart
class OCRReviewScreen extends StatefulWidget {
  final String uploadId;
  
  @override
  _OCRReviewScreenState createState() => _OCRReviewScreenState();
}

class _OCRReviewScreenState extends State<OCRReviewScreen> {
  late Future<OCRReviewData> reviewData;
  List<FieldCorrection> corrections = [];
  Stopwatch reviewTimer = Stopwatch();
  
  @override
  void initState() {
    super.initState();
    reviewData = OCRService.getReviewData(widget.uploadId);
    reviewTimer.start();
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('OCR結果の確認'),
        actions: [
          IconButton(
            icon: Icon(Icons.help_outline),
            onPressed: () => showHelpDialog(),
          ),
        ],
      ),
      body: FutureBuilder<OCRReviewData>(
        future: reviewData,
        builder: (context, snapshot) {
          if (snapshot.hasData) {
            return buildReviewInterface(snapshot.data!);
          }
          return Center(child: CircularProgressIndicator());
        },
      ),
      bottomNavigationBar: buildSubmitButton(),
    );
  }
}
```

#### 2. 信頼度による視覚的フィードバック
```dart
class OCRFieldWidget extends StatefulWidget {
  final ReviewItem item;
  final Function(FieldCorrection) onCorrection;
  
  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: item.needsReview ? 4 : 1,
      color: _getConfidenceColor(item.confidence),
      child: ExpansionTile(
        leading: _getConfidenceIcon(item.confidence),
        title: Text(
          item.label,
          style: TextStyle(
            fontWeight: item.needsReview ? FontWeight.bold : FontWeight.normal,
          ),
        ),
        subtitle: Text('信頼度: ${(item.confidence * 100).toStringAsFixed(1)}%'),
        children: [
          Padding(
            padding: EdgeInsets.all(16),
            child: Column(
              children: [
                // 元画像の該当部分を表示
                if (item.bbox != null)
                  Container(
                    height: 60,
                    child: CroppedImageWidget(
                      imageUrl: reviewData.originalImageUrl,
                      bbox: item.bbox!,
                    ),
                  ),
                
                SizedBox(height: 8),
                
                // テキスト入力フィールド
                TextFormField(
                  initialValue: item.ocrText,
                  decoration: InputDecoration(
                    labelText: 'OCR結果を確認・修正',
                    border: OutlineInputBorder(),
                    suffixIcon: item.suggestedCorrections?.isNotEmpty == true
                        ? PopupMenuButton<String>(
                            icon: Icon(Icons.arrow_drop_down),
                            onSelected: (value) {
                              setState(() {
                                textController.text = value;
                              });
                            },
                            itemBuilder: (context) => item.suggestedCorrections!
                                .map((suggestion) => PopupMenuItem(
                                      value: suggestion,
                                      child: Text(suggestion),
                                    ))
                                .toList(),
                          )
                        : null,
                  ),
                  onChanged: (value) => _handleTextChange(item, value),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
  
  Color _getConfidenceColor(double confidence) {
    if (confidence > 0.9) return Colors.green[50]!;
    if (confidence > 0.7) return Colors.yellow[50]!;
    return Colors.red[50]!;
  }
  
  Icon _getConfidenceIcon(double confidence) {
    if (confidence > 0.9) return Icon(Icons.check_circle, color: Colors.green);
    if (confidence > 0.7) return Icon(Icons.warning, color: Colors.orange);
    return Icon(Icons.error, color: Colors.red);
  }
}
```

#### 3. 効率的な確認フロー
```dart
class SmartReviewController {
  static List<ReviewItem> getHighPriorityItems(OCRReviewData data) {
    return data.reviewItems
        .where((item) => item.needsReview)
        .toList()
        ..sort((a, b) => a.confidence.compareTo(b.confidence));
  }
  
  static bool shouldShowItem(ReviewItem item, ReviewMode mode) {
    switch (mode) {
      case ReviewMode.errorsOnly:
        return item.confidence < 0.7;
      case ReviewMode.uncertainOnly:
        return item.confidence < 0.9;
      case ReviewMode.all:
        return true;
    }
  }
}

enum ReviewMode {
  errorsOnly,   // 信頼度70%未満のみ
  uncertainOnly, // 信頼度90%未満のみ
  all           // 全項目
}
```

#### 4. データ送信とフィードバック
```dart
class OCRService {
  static Future<void> submitCorrections(
    String uploadId,
    List<FieldCorrection> corrections,
    int reviewTimeSeconds,
  ) async {
    final request = {
      'upload_id': uploadId,
      'corrections': corrections.map((c) => c.toJson()).toList(),
      'review_time_seconds': reviewTimeSeconds,
      'user_feedback': {
        'overall_satisfaction': UserFeedback.overallSatisfaction,
        'ocr_accuracy_rating': UserFeedback.ocrAccuracyRating,
        'ui_usability_rating': UserFeedback.uiUsabilityRating,
      },
    };
    
    final response = await apiClient.post(
      '/api/ocr/submit-corrections',
      data: request,
    );
    
    if (response.data['triggers_retraining']) {
      // OCRモデル改善のお知らせを表示
      showRetrainingNotification();
    }
  }
}
```

## 今後の拡張計画

### Phase 1: 基本機能（MVP）
- 画像アップロード・処理
- 基本OCR（記入者番号・文字のみ）
- Flutter側OCR確認・訂正UI
- データ管理API
- 機械学習データセット生成

### Phase 2: OCR精度向上
- コメント欄OCR追加
- OCR学習データ収集システム
- 自動再学習機能
- 信頼度による処理分岐
- ユーザーフィードバック統合

### Phase 3: 高度機能
- AI評価システム統合
- リアルタイム処理
- バッチ処理機能
- 自動データ拡張
- カスタムOCRモデル最適化

### Phase 4: クラウド対応・MLOps
- AWS/GCP統合
- スケーラブル処理
- 分散ストレージ
- モデル学習パイプライン
- A/Bテスト機能
- パフォーマンス監視・アラート