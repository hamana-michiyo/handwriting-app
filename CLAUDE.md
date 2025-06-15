# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Python API Server (handwriting-eval-api)

#### Running the Server
```bash
cd handwriting-eval-api
# Development server with auto-reload
uvicorn api_server:app --reload --host 0.0.0.0 --port 8001

# Access Swagger UI documentation
# http://localhost:8001/docs
```

#### CLI Evaluation Tool
```bash
cd handwriting-eval-api
# Basic evaluation
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg

# Enhanced analysis (Phase 1.5 features)
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg --enhanced

# JSON output format
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg --enhanced --json

# Debug mode with intermediate images
python evaluate.py data/samples/ref_光.jpg data/samples/user_光1.jpg --dbg
```

#### Testing Commands
```bash
cd handwriting-eval-api
# Unit tests
pytest
pytest tests/test_metrics.py -v

# Performance validation scripts
python validation/shape_evaluation_comparison.py
python validation/test_position_robustness.py
python validation/test_scale_robustness.py
python validation/test_enhanced_analysis.py
```

### Flutter Apps

#### Main Data Management App (moji_manage_app)
```bash
cd moji_manage_app
flutter pub get           # Install dependencies
flutter run              # Run on connected device/emulator
flutter test             # Run tests
flutter analyze          # Static analysis
flutter clean            # Clean build cache

# Platform builds
flutter build apk        # Android APK
flutter build ios        # iOS (requires macOS)
flutter build web        # Web version
```

#### Basic App (bimoji_app)
```bash
cd bimoji_app
flutter pub get && flutter run
```

## Project Architecture

This is a **dual-component handwriting evaluation system** with distinct but complementary roles:

### 1. Flutter Data Collection App (`moji_manage_app`)
**Architecture Pattern**: Service-oriented MVC with local file storage

- **Purpose**: Capture, manage, and organize handwriting samples for ML dataset creation
- **Core Data Model**: `CaptureData` - represents captured samples with writer ID, timestamp, character labels, processing status
- **Key Service**: `CameraService` - handles camera operations, image capture, gallery selection
- **Storage**: Local app documents directory under `/captures/` with naming convention `記入者番号_文字_日付.jpg`

**Data Flow**:
1. `ImageCaptureScreen` → Camera/Gallery → `CameraService`
2. Image processing (planned: OpenCV distortion correction via Python backend)
3. Automatic character segmentation and storage
4. `HomeScreen` dashboard for data management

### 2. Python Evaluation Engine (`handwriting-eval-api`)
**Architecture Pattern**: Pipeline-based evaluation with modular scoring

#### Core Pipeline (`src/eval/pipeline.py`)
```python
# Main evaluation flow
preprocess_image() → evaluate_4_axes() → weighted_scoring() → detailed_diagnostics()
```

#### 4-Axis Evaluation System
- **形 (Shape)**: Multi-scale Position-corrected IoU (70%) + Improved Hu Moments (30%)
- **黒 (Black)**: Line width stability (60%) + Intensity uniformity (40%) 
- **白 (White)**: Black pixel density similarity via Gaussian evaluation
- **場 (Center)**: Character positioning based on centroid distance

#### Advanced Features (Enhanced Mode)
- **Local Texture Analysis**: 15x15 sliding window for local density variations
- **Edge Strength Evaluation**: Sobel filters for boundary clarity assessment
- **Multi-threshold Pressure Estimation**: Histogram analysis for brush pressure
- **Directional Stroke Analysis**: Separate evaluation for vertical/horizontal/diagonal strokes

### 3. FastAPI Integration Layer
**RESTful API** with CORS support for frontend integration:
- `POST /evaluate` - Base64 image evaluation
- `POST /evaluate/upload` - File upload evaluation  
- `GET /docs` - Swagger UI documentation
- Automatic BGR↔RGB conversion for OpenCV compatibility

## Technical Implementation Details

### Image Processing Pipeline
1. **Preprocessing**: Grayscale conversion → Trapezoid correction (Hough transform) → Otsu binarization
2. **Shape Evaluation**: Multi-scale template matching across factors [0.5, 0.7, 0.8, 1.0, 1.2, 1.4, 2.0]
3. **Line Quality**: Distance transform for width estimation + Grayscale analysis for intensity uniformity
4. **Enhanced Analysis**: Selective integration (Basic 85% + Enhanced Intensity 15% + Improved Width 10%)

### Performance Characteristics
- **Position Robustness**: 100% score retention for identical shapes with position offset
- **Scale Robustness**: 89%+ accuracy for similar shapes at different sizes
- **Shape Discrimination**: Appropriate differentiation between different shapes (circle vs square: 86.7%)

### Key Configuration Parameters
```python
# Scoring weights
SHAPE_W = 0.30, BLACK_W = 0.20, WHITE_W = 0.30, CENTER_W = 0.20

# Shape evaluation
SHAPE_IOU_WEIGHT = 0.7, SHAPE_HU_WEIGHT = 0.3
SCALE_FACTORS = [0.5, 0.7, 0.8, 1.0, 1.2, 1.4, 2.0]

# Line quality evaluation  
BLACK_WIDTH_WEIGHT = 0.6, BLACK_INTENSITY_WEIGHT = 0.4
```

## Development Workflow

### Adding New Evaluation Features
1. Implement core algorithm in `src/eval/metrics.py`
2. Add unit tests in `tests/test_metrics.py`
3. Create validation script in `validation/` directory
4. Update pipeline in `src/eval/pipeline.py`
5. Add CLI option in `src/eval/cli.py`
6. Update API endpoints in `api_server.py`

### Testing Strategy
- **Unit Tests**: Module-level functionality testing with pytest
- **Integration Tests**: End-to-end pipeline validation  
- **Performance Validation**: Robustness testing with known image pairs
- **API Testing**: HTTP endpoint validation with sample data

### Flutter App Development
- **Camera Integration**: Use `CameraService` for consistent image capture across screens
- **Data Management**: Follow `CaptureData` model pattern for all handwriting samples
- **Local Storage**: Images stored in app documents with consistent naming convention
- **Future API Integration**: Design for HTTP calls to Python evaluation backend

## Critical Development Notes

### Image Format Handling
- OpenCV uses BGR, web/mobile typically use RGB - automatic conversion implemented in API
- All evaluation expects 8-bit grayscale or BGR color images
- Binary masks use 0 (background) and 255 (foreground)

### Japanese UI Context
The Flutter app uses Japanese throughout as it targets Japanese handwriting evaluation:
- 記入者番号 = Writer ID/Number
- 美文字 = Beautiful handwriting  
- 評価 = Evaluation
- 形・黒・白・場 = Shape, Black (ink), White (spacing), Center (positioning)

### Dev Container Support
Both components support VSCode Dev Container development:
- Python environment with all dependencies pre-installed
- Flutter SDK and platform tools configured
- Jupyter notebook support for experimental development