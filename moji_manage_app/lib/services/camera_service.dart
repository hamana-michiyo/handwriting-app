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

      _controller = CameraController(
        cameras.first,
        ResolutionPreset.high,
        enableAudio: false,
      );

      await _controller!.initialize();
      return true;
    } catch (e) {
      print('カメラの初期化に失敗: $e');
      return false;
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
        imageQuality: 90,
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