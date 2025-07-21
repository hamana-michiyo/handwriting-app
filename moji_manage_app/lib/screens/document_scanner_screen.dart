import 'dart:io';
import 'package:flutter/material.dart';
import 'package:cunning_document_scanner/cunning_document_scanner.dart';
import 'image_preview_screen.dart';

/// Cunning Document Scanner実験画面
class DocumentScannerScreen extends StatefulWidget {
  const DocumentScannerScreen({super.key});

  @override
  State<DocumentScannerScreen> createState() => _DocumentScannerScreenState();
}

class _DocumentScannerScreenState extends State<DocumentScannerScreen> {
  List<String> _scannedImages = [];
  bool _isScanning = false;

  /// 基本スキャン（自動エッジ検出 + 透視変換）
  Future<void> _scanBasic() async {
    setState(() => _isScanning = true);
    
    try {
      List<String> pictures = await CunningDocumentScanner.getPictures() ?? [];
      if (pictures.isNotEmpty && mounted) {
        setState(() {
          _scannedImages.addAll(pictures);
        });
        _showResultDialog('基本スキャン完了', '${pictures.length}枚の画像を取得しました');
      }
    } catch (e) {
      _showErrorDialog('基本スキャン失敗', e.toString());
    } finally {
      setState(() => _isScanning = false);
    }
  }

  /// 高品質スキャン（全フィルタ適用）
  Future<void> _scanHighQuality() async {
    setState(() => _isScanning = true);
    
    try {
      List<String> pictures = await CunningDocumentScanner.getPictures(
        isGalleryImportAllowed: true,  // ギャラリー選択許可
        noOfPages: 5,                  // 最大5ページ
      ) ?? [];
      
      if (pictures.isNotEmpty && mounted) {
        setState(() {
          _scannedImages.addAll(pictures);
        });
        _showResultDialog('高品質スキャン完了', '${pictures.length}枚の高品質画像を取得しました');
      }
    } catch (e) {
      _showErrorDialog('高品質スキャン失敗', e.toString());
    } finally {
      setState(() => _isScanning = false);
    }
  }

  /// ギャラリーから選択してスキャン
  Future<void> _scanFromGallery() async {
    setState(() => _isScanning = true);
    
    try {
      List<String> pictures = await CunningDocumentScanner.getPictures(
        isGalleryImportAllowed: true,
        // v1.2.3ではsourceパラメータは使用できないため、isGalleryImportAllowedのみ使用
      ) ?? [];
      
      if (pictures.isNotEmpty && mounted) {
        setState(() {
          _scannedImages.addAll(pictures);
        });
        _showResultDialog('ギャラリースキャン完了', '${pictures.length}枚の画像を処理しました');
      }
    } catch (e) {
      _showErrorDialog('ギャラリースキャン失敗', e.toString());
    } finally {
      setState(() => _isScanning = false);
    }
  }

  /// カメラのみでスキャン（基本スキャンと同じ）
  Future<void> _scanCameraOnly() async {
    setState(() => _isScanning = true);
    
    try {
      List<String> pictures = await CunningDocumentScanner.getPictures(
        noOfPages: 1,  // 1ページのみ
      ) ?? [];
      
      if (pictures.isNotEmpty && mounted) {
        setState(() {
          _scannedImages.addAll(pictures);
        });
        _showResultDialog('カメラスキャン完了', '${pictures.length}枚の画像を取得しました');
      }
    } catch (e) {
      _showErrorDialog('カメラスキャン失敗', e.toString());
    } finally {
      setState(() => _isScanning = false);
    }
  }

  /// 結果表示ダイアログ
  void _showResultDialog(String title, String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            const Icon(Icons.check_circle, color: Colors.green),
            const SizedBox(width: 8),
            Text(title),
          ],
        ),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  /// エラー表示ダイアログ
  void _showErrorDialog(String title, String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            const Icon(Icons.error, color: Colors.red),
            const SizedBox(width: 8),
            Text(title),
          ],
        ),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  /// 画像をプレビュー画面で表示
  void _previewImage(String imagePath) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ImagePreviewScreen(imagePath: imagePath),
      ),
    );
  }

  /// 全画像クリア
  void _clearImages() {
    setState(() {
      _scannedImages.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('📄 Document Scanner実験'),
        backgroundColor: Colors.purple.shade50,
        actions: [
          if (_scannedImages.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.clear_all),
              onPressed: _clearImages,
              tooltip: '全クリア',
            ),
        ],
      ),
      body: _isScanning
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('スキャン中...'),
                ],
              ),
            )
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildScanOptions(),
                  const SizedBox(height: 24),
                  _buildResults(),
                ],
              ),
            ),
    );
  }

  /// スキャンオプション
  Widget _buildScanOptions() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '🔍 スキャンオプション',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            
            // 基本スキャン
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: _scanBasic,
                icon: const Icon(Icons.document_scanner),
                label: const Text('基本スキャン'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 12),
                ),
              ),
            ),
            const Text(
              '自動エッジ検出 + 透視変換のベーシック機能',
              style: TextStyle(color: Colors.grey, fontSize: 12),
            ),
            
            const SizedBox(height: 12),
            
            // 高品質スキャン
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: _scanHighQuality,
                icon: const Icon(Icons.high_quality),
                label: const Text('高品質スキャン'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 12),
                ),
              ),
            ),
            const Text(
              '画質向上フィルタ + 複数ページ対応（最大5ページ）',
              style: TextStyle(color: Colors.grey, fontSize: 12),
            ),
            
            const SizedBox(height: 12),
            
            // ギャラリースキャン
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: _scanFromGallery,
                icon: const Icon(Icons.photo_library),
                label: const Text('ギャラリーから'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.orange,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 12),
                ),
              ),
            ),
            const Text(
              'ギャラリーの画像に自動エッジ検出・透視変換を適用',
              style: TextStyle(color: Colors.grey, fontSize: 12),
            ),
            
            const SizedBox(height: 12),
            
            // カメラのみ
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: _scanCameraOnly,
                icon: const Icon(Icons.camera_alt),
                label: const Text('カメラのみ'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.purple,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 12),
                ),
              ),
            ),
            const Text(
              'カメラ撮影のみ（1ページ限定）',
              style: TextStyle(color: Colors.grey, fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }

  /// スキャン結果表示
  Widget _buildResults() {
    if (_scannedImages.isEmpty) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Center(
            child: Column(
              children: [
                Icon(
                  Icons.document_scanner_outlined,
                  size: 64,
                  color: Colors.grey,
                ),
                SizedBox(height: 16),
                Text(
                  'スキャン結果がここに表示されます',
                  style: TextStyle(color: Colors.grey),
                ),
              ],
            ),
          ),
        ),
      );
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.image, color: Colors.blue),
                const SizedBox(width: 8),
                Text(
                  'スキャン結果 (${_scannedImages.length}枚)',
                  style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 16),
            GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                crossAxisSpacing: 8,
                mainAxisSpacing: 8,
              ),
              itemCount: _scannedImages.length,
              itemBuilder: (context, index) {
                final imagePath = _scannedImages[index];
                return GestureDetector(
                  onTap: () => _previewImage(imagePath),
                  child: Container(
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey.shade300),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.file(
                        File(imagePath),
                        fit: BoxFit.cover,
                        errorBuilder: (context, error, stackTrace) {
                          return Container(
                            color: Colors.grey.shade200,
                            child: const Icon(
                              Icons.broken_image,
                              color: Colors.grey,
                            ),
                          );
                        },
                      ),
                    ),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}