import 'package:flutter/material.dart';
import '../services/camera_service.dart';
import 'fullscreen_camera_screen.dart';
import 'image_preview_screen.dart';

class ImageCaptureScreen extends StatefulWidget {
  const ImageCaptureScreen({super.key});

  @override
  State<ImageCaptureScreen> createState() => _ImageCaptureScreenState();
}

class _ImageCaptureScreenState extends State<ImageCaptureScreen> {
  final CameraService _cameraService = CameraService();
  
  bool _isCameraInitialized = false;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    setState(() => _isLoading = true);
    final success = await _cameraService.initializeCamera();
    setState(() {
      _isCameraInitialized = success;
      _isLoading = false;
    });
  }

  Future<void> _takePicture() async {
    if (!_isCameraInitialized) return;
    
    // フルスクリーンカメラ画面を開く
    final String? imagePath = await Navigator.push<String?>(
      context,
      MaterialPageRoute(
        builder: (context) => FullscreenCameraScreen(
          cameraService: _cameraService,
        ),
      ),
    );
    
    if (imagePath != null && mounted) {
      // プレビュー画面を表示
      final bool? result = await Navigator.push<bool?>(
        context,
        MaterialPageRoute(
          builder: (context) => ImagePreviewScreen(
            imagePath: imagePath,
          ),
        ),
      );
      
      // アップロード成功時は撮影画面に戻る
      if (result == true && mounted) {
        Navigator.pop(context); // 撮影画面を閉じる
      }
    }
  }

  Future<void> _pickFromGallery() async {
    setState(() => _isLoading = true);
    final imagePath = await _cameraService.pickImageFromGallery();
    
    if (imagePath != null && mounted) {
      // プレビュー画面を表示
      final bool? result = await Navigator.push<bool?>(
        context,
        MaterialPageRoute(
          builder: (context) => ImagePreviewScreen(
            imagePath: imagePath,
          ),
        ),
      );
      
      // アップロード成功時は撮影画面に戻る
      if (result == true && mounted) {
        Navigator.pop(context); // 撮影画面を閉じる
      }
    }
    setState(() => _isLoading = false);
  }

  @override
  void dispose() {
    _cameraService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('📷 画像取り込み'),
        backgroundColor: Colors.blue.shade50,
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildCaptureMethodSelection(),
                ],
              ),
            ),
    );
  }


  Widget _buildCaptureMethodSelection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '📸 撮影方法選択',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isCameraInitialized ? _takePicture : null,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('カメラで撮影'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _pickFromGallery,
                    icon: const Icon(Icons.photo_library),
                    label: const Text('ギャラリーから選択'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

}