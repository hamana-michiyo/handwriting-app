import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:image/image.dart' as img;
import 'image_upload_screen.dart';

class ImagePreviewScreen extends StatefulWidget {
  final String imagePath;

  const ImagePreviewScreen({
    super.key,
    required this.imagePath,
  });

  @override
  State<ImagePreviewScreen> createState() => _ImagePreviewScreenState();
}

class _ImagePreviewScreenState extends State<ImagePreviewScreen> {
  String? _croppedImagePath;
  bool _isCropping = false;

  String get _currentImagePath => _croppedImagePath ?? widget.imagePath;

  /// 画像を切り取る
  Future<void> _cropImage() async {
    setState(() {
      _isCropping = true;
    });

    try {
      final croppedFile = await ImageCropper().cropImage(
        sourcePath: widget.imagePath,
        maxWidth: 4000,
        maxHeight: 3000,
        compressQuality: 100,
        uiSettings: [
          AndroidUiSettings(
            toolbarTitle: '📝 記入用紙を切り取る',
            toolbarColor: Colors.blue,
            toolbarWidgetColor: Colors.white,
            statusBarColor: Colors.blue.shade700,
            backgroundColor: Colors.black,
            activeControlsWidgetColor: Colors.blue,
            dimmedLayerColor: Colors.black.withOpacity(0.5),
            cropFrameColor: Colors.blue,
            cropGridColor: Colors.blue.withOpacity(0.5),
            cropFrameStrokeWidth: 3,
            cropGridStrokeWidth: 1,
            showCropGrid: true,
            lockAspectRatio: false,
            hideBottomControls: false,
          ),
          IOSUiSettings(
            title: '📝 記入用紙を切り取る',
            minimumAspectRatio: 0.5,
            aspectRatioLockDimensionSwapEnabled: false,
            aspectRatioLockEnabled: false,
          ),
        ],
      );

      if (croppedFile != null) {
        setState(() {
          _croppedImagePath = croppedFile.path;
        });

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('✂️ 画像を切り取りました！次はサーバーで自動正面化します'),
              backgroundColor: Colors.green,
              duration: Duration(seconds: 3),
            ),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('切り取りに失敗しました: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      setState(() {
        _isCropping = false;
      });
    }
  }

  /// アップロード画面に遷移
  Future<void> _proceedToUpload(BuildContext context) async {
    final result = await Navigator.push<bool>(
      context,
      MaterialPageRoute(
        builder: (context) => ImageUploadScreen(imagePath: _currentImagePath),
      ),
    );
    
    if (result == true && context.mounted) {
      // アップロード成功時は2つ前の画面まで戻る（撮影画面まで）
      Navigator.pop(context, true);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
        title: Text(
          _croppedImagePath != null ? '✂️ 切り取り済み' : '撮影プレビュー',
          style: const TextStyle(color: Colors.white),
        ),
        actions: [
          if (_croppedImagePath != null)
            IconButton(
              onPressed: () {
                setState(() {
                  _croppedImagePath = null;
                });
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('元の画像に戻しました'),
                    backgroundColor: Colors.blue,
                  ),
                );
              },
              icon: const Icon(Icons.restore, color: Colors.white),
              tooltip: '元に戻す',
            ),
        ],
      ),
      body: _isCropping
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(color: Colors.white),
                  SizedBox(height: 16),
                  Text(
                    '✂️ 画像を切り取り中...',
                    style: TextStyle(color: Colors.white),
                  ),
                ],
              ),
            )
          : Center(
              child: Image.file(
                File(_currentImagePath),
                fit: BoxFit.contain,
                width: double.infinity,
                height: double.infinity,
              ),
            ),
      bottomNavigationBar: Container(
        color: Colors.black,
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(context).padding.bottom + 16,
          left: 16,
          right: 16,
          top: 16,
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // 切り取りボタン
            if (_croppedImagePath == null && !_isCropping)
              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: _cropImage,
                  icon: const Icon(Icons.crop, color: Colors.orange),
                  label: const Text(
                    '✂️ 記入用紙を切り取る',
                    style: TextStyle(color: Colors.orange),
                  ),
                  style: OutlinedButton.styleFrom(
                    side: const BorderSide(color: Colors.orange),
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                ),
              ),
            
            if (_croppedImagePath == null && !_isCropping)
              const SizedBox(height: 12),
            
            // 切り取り後の説明
            if (_croppedImagePath != null)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                margin: const EdgeInsets.only(bottom: 12),
                decoration: BoxDecoration(
                  color: Colors.green.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.green),
                ),
                child: const Column(
                  children: [
                    Row(
                      children: [
                        Icon(Icons.check_circle, color: Colors.green, size: 20),
                        SizedBox(width: 8),
                        Text(
                          '✂️ 切り取り完了',
                          style: TextStyle(
                            color: Colors.green,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: 4),
                    Text(
                      'サーバーで自動的に四隅検出→正面化→文字認識を行います',
                      style: TextStyle(color: Colors.green, fontSize: 12),
                    ),
                  ],
                ),
              ),
            
            // メインボタン
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _isCropping ? null : () => Navigator.pop(context, false),
                    icon: const Icon(Icons.refresh, color: Colors.white),
                    label: const Text(
                      '再撮影',
                      style: TextStyle(color: Colors.white),
                    ),
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: Colors.white),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isCropping ? null : () => _proceedToUpload(context),
                    icon: Icon(
                      _croppedImagePath != null ? Icons.auto_fix_high : Icons.cloud_upload,
                      color: Colors.black,
                    ),
                    label: Text(
                      _croppedImagePath != null ? 'AI処理' : 'アップロード',
                      style: const TextStyle(color: Colors.black),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _croppedImagePath != null ? Colors.green : Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 16),
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