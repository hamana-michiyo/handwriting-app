import 'dart:io';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

class CameraService {
  CameraController? _controller;
  final ImagePicker _picker = ImagePicker();

  Future<List<CameraDescription>> getAvailableCameras() async {
    return await availableCameras();
  }

  Future<bool> initializeCamera() async {
    try {
      final cameras = await getAvailableCameras();
      if (cameras.isEmpty) return false;

      // デバイスに応じた解像度設定
      ResolutionPreset resolution = _getOptimalResolution();

      _controller = CameraController(
        cameras.first,
        resolution,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _controller!.initialize();
      print('カメラ初期化完了 - 解像度: $resolution, アスペクト比: ${_controller!.value.aspectRatio}');
      return true;
    } catch (e) {
      print('カメラの初期化に失敗: $e');
      return false;
    }
  }

  ResolutionPreset _getOptimalResolution() {
    // プラットフォーム別に最適な解像度を選択
    if (Platform.isAndroid) {
      // Androidデバイス向け（機種の多様性を考慮）
      return ResolutionPreset.veryHigh; // 1080p
    } else if (Platform.isIOS) {
      // iOSデバイス向け（比較的統一された性能）
      return ResolutionPreset.max; // 4K
    } else {
      // その他のプラットフォーム
      return ResolutionPreset.high; // 720p
    }
  }

  CameraController? get controller => _controller;

  Future<String?> takePicture() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      return null;
    }

    try {
      final XFile picture = await _controller!.takePicture();
      final Directory appDir = await getApplicationDocumentsDirectory();
      final String fileName = 'capture_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final String filePath = path.join(appDir.path, 'captures', fileName);
      
      await Directory(path.dirname(filePath)).create(recursive: true);
      await picture.saveTo(filePath);
      
      return filePath;
    } catch (e) {
      print('写真撮影に失敗: $e');
      return null;
    }
  }

  Future<String?> pickImageFromGallery() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 100,
        maxWidth: 6000,
        maxHeight: 4000,
      );
      
      if (image == null) return null;

      final Directory appDir = await getApplicationDocumentsDirectory();
      final String fileName = 'gallery_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final String filePath = path.join(appDir.path, 'captures', fileName);
      
      await Directory(path.dirname(filePath)).create(recursive: true);
      await File(image.path).copy(filePath);
      
      return filePath;
    } catch (e) {
      print('ギャラリーからの画像選択に失敗: $e');
      return null;
    }
  }

  void dispose() {
    _controller?.dispose();
  }
}