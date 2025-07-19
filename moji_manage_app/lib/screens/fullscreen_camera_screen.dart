import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/camera_service.dart';

class FullscreenCameraScreen extends StatefulWidget {
  final CameraService cameraService;

  const FullscreenCameraScreen({
    super.key,
    required this.cameraService,
  });

  @override
  State<FullscreenCameraScreen> createState() => _FullscreenCameraScreenState();
}

class _FullscreenCameraScreenState extends State<FullscreenCameraScreen> {
  bool _isCapturing = false;

  Future<void> _takePicture() async {
    if (_isCapturing) return;

    setState(() => _isCapturing = true);
    
    final imagePath = await widget.cameraService.takePicture();
    
    if (imagePath != null && mounted) {
      Navigator.pop(context, imagePath);
    }
    
    setState(() => _isCapturing = false);
  }

  Widget _buildOptimizedCameraPreview() {
    final controller = widget.cameraService.controller!;
    final screenSize = MediaQuery.of(context).size;
    
    // カメラのアスペクト比（通常は横向き基準）
    final cameraAspectRatio = controller.value.aspectRatio;
    // 縦向きスマホの画面アスペクト比
    final screenAspectRatio = screenSize.width / screenSize.height;
    
    // スマホ縦向きに合わせてカメラのアスペクト比を逆転
    final adjustedCameraAspectRatio = 1.0 / cameraAspectRatio;
    
    print('カメラ元アスペクト比: $cameraAspectRatio (横向き基準)');
    print('調整後アスペクト比: $adjustedCameraAspectRatio (縦向き)');
    print('画面アスペクト比: $screenAspectRatio');

    // 調整後のアスペクト比で比較
    if ((adjustedCameraAspectRatio - screenAspectRatio).abs() < 0.1) {
      // アスペクト比がほぼ同じ場合は全画面表示
      return SizedBox.expand(
        child: CameraPreview(controller),
      );
    } else {
      // アスペクト比が異なる場合は正しい比率で表示
      return Center(
        child: AspectRatio(
          aspectRatio: adjustedCameraAspectRatio,
          child: CameraPreview(controller),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // カメラプレビュー（デバイス最適化）
          Positioned.fill(
            child: widget.cameraService.controller != null &&
                    widget.cameraService.controller!.value.isInitialized
                ? _buildOptimizedCameraPreview()
                : const Center(
                    child: CircularProgressIndicator(color: Colors.white),
                  ),
          ),
          
          // 上部：戻るボタン
          Positioned(
            top: MediaQuery.of(context).padding.top + 16,
            left: 16,
            child: GestureDetector(
              onTap: () => Navigator.pop(context),
              child: Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.5),
                  shape: BoxShape.circle,
                ),
                child: const Icon(
                  Icons.arrow_back,
                  color: Colors.white,
                  size: 24,
                ),
              ),
            ),
          ),
          
          // 下部：シャッターボタン
          Positioned(
            bottom: MediaQuery.of(context).padding.bottom + 40,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTap: _takePicture,
                child: Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.white,
                    border: Border.all(
                      color: Colors.white,
                      width: 4,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.3),
                        blurRadius: 10,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: _isCapturing
                      ? const Center(
                          child: CircularProgressIndicator(
                            color: Colors.grey,
                            strokeWidth: 3,
                          ),
                        )
                      : Container(
                          margin: const EdgeInsets.all(6),
                          decoration: const BoxDecoration(
                            shape: BoxShape.circle,
                            color: Colors.white,
                          ),
                        ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}