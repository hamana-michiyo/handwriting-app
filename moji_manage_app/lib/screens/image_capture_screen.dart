import 'dart:io';
import 'package:flutter/material.dart';
import '../services/camera_service.dart';
import '../models/capture_data.dart';
import 'fullscreen_camera_screen.dart';
import 'image_preview_screen.dart';

class ImageCaptureScreen extends StatefulWidget {
  const ImageCaptureScreen({super.key});

  @override
  State<ImageCaptureScreen> createState() => _ImageCaptureScreenState();
}

class _ImageCaptureScreenState extends State<ImageCaptureScreen> {
  final CameraService _cameraService = CameraService();
  final TextEditingController _writerNumberController = TextEditingController();
  
  bool _isCameraInitialized = false;
  bool _isLoading = false;
  String? _capturedImagePath;
  bool _showPreview = false;

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
      final bool? useImage = await Navigator.push<bool?>(
        context,
        MaterialPageRoute(
          builder: (context) => ImagePreviewScreen(
            imagePath: imagePath,
          ),
        ),
      );
      
      if (useImage == true && mounted) {
        setState(() {
          _capturedImagePath = imagePath;
          _showPreview = true;
        });
      }
    }
  }

  Future<void> _pickFromGallery() async {
    setState(() => _isLoading = true);
    final imagePath = await _cameraService.pickImageFromGallery();
    
    if (imagePath != null) {
      setState(() {
        _capturedImagePath = imagePath;
        _showPreview = true;
      });
    }
    setState(() => _isLoading = false);
  }

  void _retake() {
    setState(() {
      _capturedImagePath = null;
      _showPreview = false;
    });
  }

  void _proceedToNext() {
    if (_capturedImagePath == null || _writerNumberController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('記入者番号と画像が必要です')),
      );
      return;
    }

    final captureData = CaptureData(
      imagePath: _capturedImagePath,
      writerNumber: _writerNumberController.text,
      captureTime: DateTime.now(),
    );

    Navigator.pop(context, captureData);
  }

  @override
  void dispose() {
    _cameraService.dispose();
    _writerNumberController.dispose();
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
                  _buildWriterNumberInput(),
                  const SizedBox(height: 20),
                  _buildCaptureMethodSelection(),
                  const SizedBox(height: 20),
                  _buildPreviewArea(),
                  const SizedBox(height: 20),
                  _buildActionButtons(),
                ],
              ),
            ),
    );
  }

  Widget _buildWriterNumberInput() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '記入者番号',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _writerNumberController,
              decoration: const InputDecoration(
                hintText: 'S001',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.person),
              ),
            ),
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

  Widget _buildPreviewArea() {
    return Card(
      child: Container(
        width: double.infinity,
        height: 400, // 高さを300から400に増加
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            const Text(
              '📷 撮影プレビュー',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade300),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: _showPreview && _capturedImagePath != null
                    ? ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.file(
                          File(_capturedImagePath!),
                          fit: BoxFit.contain,
                        ),
                      )
                    : const Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.camera_alt, size: 64, color: Colors.grey),
                            SizedBox(height: 8),
                            Text(
                              'カメラで撮影ボタンを押すと\n全画面カメラが開きます',
                              textAlign: TextAlign.center,
                              style: TextStyle(color: Colors.grey),
                            ),
                          ],
                        ),
                      ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionButtons() {
    if (!_showPreview) return const SizedBox.shrink();
    
    return Row(
      children: [
        Expanded(
          child: OutlinedButton.icon(
            onPressed: _retake,
            icon: const Icon(Icons.refresh),
            label: const Text('❌ 再撮影'),
            style: OutlinedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 12),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _proceedToNext,
            icon: const Icon(Icons.arrow_forward),
            label: const Text('✅ 次へ'),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 12),
            ),
          ),
        ),
      ],
    );
  }
}