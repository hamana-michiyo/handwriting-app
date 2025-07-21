import 'package:flutter/material.dart';
import 'package:cunning_document_scanner/cunning_document_scanner.dart';
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

  /// カメラで撮影（従来方式：camera + image_cropper）
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

  /// 自動スキャン（cunning_document_scanner方式）
  Future<void> _scanDocument() async {
    setState(() => _isLoading = true);
    
    try {
      List<String> pictures = await CunningDocumentScanner.getPictures() ?? [];
      
      if (pictures.isNotEmpty && mounted) {
        // 最初の画像をプレビュー画面で表示
        final bool? result = await Navigator.push<bool?>(
          context,
          MaterialPageRoute(
            builder: (context) => ImagePreviewScreen(
              imagePath: pictures.first,
            ),
          ),
        );
        
        // アップロード成功時は撮影画面に戻る
        if (result == true && mounted) {
          Navigator.pop(context); // 撮影画面を閉じる
        }
      } else if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('スキャンがキャンセルされました'),
            backgroundColor: Colors.orange,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('スキャンに失敗しました: ${e.toString()}'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
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
        title: const Text('画像取り込み'),
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
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          '撮影方法選択',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 20),
        
        // カメラで撮影ボタン
        SizedBox(
          width: double.infinity,
          height: 80,
          child: ElevatedButton.icon(
            onPressed: _isCameraInitialized ? _takePicture : null,
            icon: const Icon(Icons.camera_alt, size: 28),
            label: const Text('カメラで撮影', style: TextStyle(fontSize: 18)),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.lightBlue.shade200,
              foregroundColor: Colors.black87,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              elevation: 2,
            ),
          ),
        ),
        
        const SizedBox(height: 16),
        
        // 自動スキャンボタン
        SizedBox(
          width: double.infinity,
          height: 80,
          child: ElevatedButton.icon(
            onPressed: _scanDocument,
            icon: const Icon(Icons.document_scanner, size: 28),
            label: const Text('自動スキャン', style: TextStyle(fontSize: 18)),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.lightGreen.shade200,
              foregroundColor: Colors.black87,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              elevation: 2,
            ),
          ),
        ),
        
        const SizedBox(height: 24),
        
        // 使い方のヒント
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.grey.shade100,
            borderRadius: BorderRadius.circular(12),
          ),
          child: const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '使い方のヒント',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 8),
              Text(
                '• カメラで撮影：従来通りの撮影→クロップ方式\n'
                '• 自動スキャン：エッジ検出→透視変換で高品質',
                style: TextStyle(fontSize: 14, color: Colors.black87),
              ),
            ],
          ),
        ),
        
        const SizedBox(height: 24),
        
        // ギャラリーから選択ボタン
        SizedBox(
          width: double.infinity,
          height: 60,
          child: OutlinedButton.icon(
            onPressed: _pickFromGallery,
            icon: const Icon(Icons.photo_library, size: 24),
            label: const Text('ギャラリーから選択', style: TextStyle(fontSize: 16)),
            style: OutlinedButton.styleFrom(
              foregroundColor: Colors.purple,
              side: const BorderSide(color: Colors.purple, width: 2),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),
        ),
      ],
    );
  }
}