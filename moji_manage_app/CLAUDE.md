# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Flutter Commands
- `flutter run` - Run the app on connected device/emulator
- `flutter build apk` - Build Android APK
- `flutter build ios` - Build iOS app
- `flutter test` - Run all tests
- `flutter analyze` - Analyze code for issues
- `flutter clean` - Clean build cache and dependencies
- `flutter pub get` - Install dependencies from pubspec.yaml
- `flutter pub upgrade` - Upgrade dependencies

### Platform-specific Building
- Android: Use `flutter build apk` or `flutter build appbundle`
- iOS: Use `flutter build ios` (requires macOS and Xcode)
- Desktop: `flutter build windows|linux|macos`

## Project Architecture

This is a Flutter app for managing handwriting evaluation data for machine learning purposes. The app is designed to capture, process, and evaluate handwritten characters for a Japanese calligraphy ("美文字") assessment system.

### Core Application Flow
1. **Image Capture** (`ImageCaptureScreen`) - Users capture/select images of handwritten characters with registration marks (トンボ) for distortion correction
2. **Data Management** (`HomeScreen`) - Dashboard showing statistics and recent captures
3. **Future Evaluation** - Planned evaluation system for scoring characters on 4 criteria (形/黒/白/場)

### Key Data Model
- `CaptureData` - Represents captured handwriting samples with writer ID, timestamp, character labels, and processing status
- Images stored locally with naming convention: `記入者番号_文字_日付.jpg`

### Services Architecture
- `CameraService` - Handles camera initialization, photo capture, and gallery selection
- All captured images saved to app documents directory under `/captures/`

### Technical Stack
- **Frontend**: Flutter with Material Design
- **Camera**: `camera` package for live preview and capture
- **Image Handling**: `image_picker` for gallery selection
- **Storage**: Local file system using `path_provider`
- **Future Backend**: Planned Python/OpenCV integration for automatic character segmentation and distortion correction

### Japanese UI Context
The app uses Japanese text throughout as it's designed for Japanese handwriting evaluation. Key terms:
- 記入者番号 = Writer ID/Number
- 美文字 = Beautiful handwriting
- 評価 = Evaluation
- The 4 evaluation criteria are: 形 (shape), 黒 (ink strength), 白 (spacing), 場 (positioning)

### Image Processing Pipeline (Planned)
1. Detect registration marks (トンボ) in captured images
2. Apply distortion correction using OpenCV
3. Automatically segment individual character cells
4. Save processed character images for evaluation