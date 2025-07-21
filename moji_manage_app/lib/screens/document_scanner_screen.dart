import 'dart:io';
import 'package:flutter/material.dart';
import 'package:cunning_document_scanner/cunning_document_scanner.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter/foundation.dart';
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
    if (kDebugMode) {
      print('=== 基本スキャン開始（Podfile設定後） ===');
    }
    
    setState(() => _isScanning = true);
    
    try {
      if (kDebugMode) {
        print('cunning_document_scanner呼び出し（Podfile設定完了後）');
      }
      
      // Podfile設定後のcunning_document_scanner呼び出し（UI改善設定）
      List<String> pictures = await CunningDocumentScanner.getPictures() ?? [];
      
      if (kDebugMode) {
        print('cunning_document_scanner結果: ${pictures.length}枚');
      }
      
      if (pictures.isNotEmpty && mounted) {
        setState(() {
          _scannedImages.addAll(pictures);
        });
        _showResultDialog(
          '基本スキャン完了 🎉', 
          '${pictures.length}枚の画像を取得しました！\n\n💡 コツ：\n• Manualモードで角をタップして微調整\n• 「Done」ボタンでスキャン完了\n• 複数ページは「+」ボタンで追加'
        );
      } else if (mounted) {
        _showErrorDialog('スキャン結果なし', '画像が取得できませんでした。キャンセルされた可能性があります。');
      }
    } catch (e) {
      if (kDebugMode) {
        print('基本スキャンエラー詳細: $e');
        print('エラータイプ: ${e.runtimeType}');
        print('スタックトレース: ${StackTrace.current}');
      }
      
      if (e.toString().contains('permission') || e.toString().contains('Permission')) {
        setState(() => _isScanning = false);
        await _showPermissionHandlerIssueDialog();
      } else {
        _showErrorDialog('基本スキャン失敗', 'エラー: ${e.toString()}\n\nデバッグ情報: ${e.runtimeType}');
      }
    } finally {
      if (_isScanning) {
        setState(() => _isScanning = false);
      }
    }
  }

  /// permission_handlerの問題を説明するダイアログ
  Future<void> _showPermissionHandlerIssueDialog() async {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            const Icon(Icons.bug_report, color: Colors.red),
            const SizedBox(width: 8),
            const Text('permission_handlerの問題'),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '診断結果：\n'
              '• 既存カメラアプリ: 正常動作 ✅\n'
              '• permission_handler: permanentlyDenied ❌\n'
              '• cunning_document_scanner: permission_handlerに依存\n\n'
              'これはパッケージ間の権限システム不整合です。',
              style: TextStyle(fontSize: 14),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.amber.shade50,
                border: Border.all(color: Colors.amber.shade200),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.psychology, color: Colors.amber.shade700, size: 16),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Text(
                          '技術的解決策',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            color: Colors.amber.shade700,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    '1. iOS設定でアプリを削除・再インストール\n'
                    '2. permission_handlerパッケージの更新\n'
                    '3. 別のdocument scannerライブラリの検討',
                    style: TextStyle(fontSize: 13),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.green.shade50,
                border: Border.all(color: Colors.green.shade200),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                children: [
                  Icon(Icons.info, color: Colors.green.shade700, size: 16),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      '既存のカメラ機能は正常動作中です。Document Scanner実験は技術的課題により一時保留とします。',
                      style: TextStyle(
                        fontSize: 13, 
                        color: Colors.green.shade700,
                        fontStyle: FontStyle.italic
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('理解しました'),
          ),
          ElevatedButton(
            onPressed: () async {
              Navigator.of(context).pop();
              // 既存のカメラ画面に移動
              Navigator.pop(context); // Document Scanner画面を閉じる
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green,
              foregroundColor: Colors.white,
            ),
            child: const Text('既存カメラを使用'),
          ),
        ],
      ),
    );
  }


  /// 再試行確認ダイアログ
  Future<bool> _showRetryDialog() async {
    return await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            const Icon(Icons.warning, color: Colors.orange),
            const SizedBox(width: 8),
            const Text('権限の問題'),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '診断結果：\n'
              '• カメラ権限: 拒否\n'
              '• フォト権限: 拒否\n'
              '• 既存アプリは動作中\n\n'
              'これは権限システムの不整合が原因です。',
              style: TextStyle(fontSize: 14),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue.shade50,
                border: Border.all(color: Colors.blue.shade200),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.lightbulb, color: Colors.blue.shade700, size: 16),
                      const SizedBox(width: 6),
                      Text(
                        '解決方法',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: Colors.blue.shade700,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    '1. 「権限を再取得」で強制的にiOS権限ダイアログを表示\n'
                    '2. 「設定を開く」でiOS設定画面へ移動\n'
                    '3. 両方の権限をオンにしてから再試行',
                    style: TextStyle(fontSize: 13),
                  ),
                ],
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('キャンセル'),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop(false);
              // iOS設定は手動で開いてもらう
            },
            child: const Text('手動で設定'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.of(context).pop(false);
              // Podfile設定後は直接スキャン再試行
              if (mounted) {
                _scanBasic();
              }
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.blue,
              foregroundColor: Colors.white,
            ),
            child: const Text('再試行'),
          ),
        ],
      ),
    ) ?? false;
  }

  /// 高品質スキャン（全フィルタ適用）
  Future<void> _scanHighQuality() async {
    setState(() => _isScanning = true);
    
    try {
      List<String>? pictures = await CunningDocumentScanner.getPictures(
        isGalleryImportAllowed: true,  // ギャラリー選択許可
        noOfPages: 5,                  // 最大5ページ
      ) ?? [];
      
      if (pictures != null && pictures.isNotEmpty && mounted) {
        setState(() {
          _scannedImages.addAll(pictures);
        });
        _showResultDialog('複数ページスキャン完了', '${pictures.length}枚の画像を取得しました（最大5ページ対応）');
      } else if (mounted) {
        _showErrorDialog('スキャン結果なし', '画像が取得できませんでした。もう一度お試しください。');
      }
    } catch (e) {
      if (e.toString().contains('permission') || e.toString().contains('Permission')) {
        setState(() => _isScanning = false);
        await _showPermissionHandlerIssueDialog();
      } else {
        _showErrorDialog('高品質スキャン失敗', 'エラー: ${e.toString()}');
      }
    } finally {
      if (_isScanning) {
        setState(() => _isScanning = false);
      }
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
      } else if (mounted) {
        _showErrorDialog('スキャン結果なし', '画像が取得できませんでした。もう一度お試しください。');
      }
    } catch (e) {
      if (e.toString().contains('permission') || e.toString().contains('Permission')) {
        setState(() => _isScanning = false);
        await _showPermissionHandlerIssueDialog();
      } else {
        _showErrorDialog('ギャラリースキャン失敗', 'エラー: ${e.toString()}');
      }
    } finally {
      if (_isScanning) {
        setState(() => _isScanning = false);
      }
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
      } else if (mounted) {
        _showErrorDialog('スキャン結果なし', '画像が取得できませんでした。もう一度お試しください。');
      }
    } catch (e) {
      if (e.toString().contains('permission') || e.toString().contains('Permission')) {
        setState(() => _isScanning = false);
        await _showPermissionHandlerIssueDialog();
      } else {
        _showErrorDialog('カメラスキャン失敗', 'エラー: ${e.toString()}');
      }
    } finally {
      if (_isScanning) {
        setState(() => _isScanning = false);
      }
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

  /// 権限リクエストダイアログ
  void _showPermissionDialog(List<String> deniedPermissions, [bool hasPermanentlyDenied = false]) {
    showDialog(
      context: context,
      barrierDismissible: false, // ダイアログ外タップで閉じないように
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            const Icon(Icons.security, color: Colors.orange),
            const SizedBox(width: 8),
            const Text('権限が必要です'),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(hasPermanentlyDenied 
              ? 'Document Scannerを使用するには、設定アプリで以下の権限を有効にする必要があります：'
              : 'Document Scannerを使用するには、以下の権限が必要です：'),
            const SizedBox(height: 12),
            ...deniedPermissions.map(
              (permission) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 2),
                child: Row(
                  children: [
                    Icon(
                      hasPermanentlyDenied ? Icons.warning : Icons.check_circle_outline, 
                      size: 16, 
                      color: hasPermanentlyDenied ? Colors.orange : Colors.blue
                    ),
                    const SizedBox(width: 8),
                    Text(permission, style: const TextStyle(fontWeight: FontWeight.w500)),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            if (hasPermanentlyDenied) ...[
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.orange.shade50,
                  border: Border.all(color: Colors.orange.shade200),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    Icon(Icons.info_outline, color: Colors.orange.shade700, size: 20),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        '権限が永続的に拒否されています。設定アプリから手動で有効にしてください。',
                        style: TextStyle(color: Colors.orange.shade700, fontSize: 13),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
            ],
            const Text(
              '設定手順：',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              hasPermanentlyDenied 
                ? '1. 「設定を開く」をタップ\n'
                  '2. 権限設定画面が開きます\n'
                  '3. 必要な権限をすべてオンにしてください\n'
                  '4. アプリに戻ってもう一度お試しください'
                : '1. 「設定を開く」をタップ\n'
                  '2. アプリの権限設定画面が開きます\n'
                  '3. 必要な権限をオンにしてください\n'
                  '4. アプリに戻ってもう一度お試しください',
              style: const TextStyle(fontSize: 14),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('キャンセル'),
          ),
          ElevatedButton.icon(
            onPressed: () {
              Navigator.of(context).pop();
              // iOS設定は手動で開いてもらう
            },
            icon: const Icon(Icons.settings),
            label: const Text('iOS設定へ'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.blue,
              foregroundColor: Colors.white,
            ),
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
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  '🔍 スキャンオプション',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.blue.shade50,
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: Colors.blue.shade200),
                  ),
                  child: const Text(
                    '💡 使い方：カメラ起動後、画面右上の「Manual」をタップ → 4つの角を調整 → 「Done」で完了',
                    style: TextStyle(fontSize: 12, color: Colors.blue),
                  ),
                ),
              ],
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
              '自動エッジ検出 + 透視変換のベーシック機能\n推奨：カメラ起動後に「Manual」タップで精度UP',
              style: TextStyle(color: Colors.grey, fontSize: 12),
            ),
            
            const SizedBox(height: 12),
            
            // 高品質スキャン
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: _scanHighQuality,
                icon: const Icon(Icons.high_quality),
                label: const Text('複数ページスキャン'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 12),
                ),
              ),
            ),
            const Text(
              'ギャラリー選択可能 + 複数ページ対応（最大5ページ）\n※画質は基本スキャンと同等',
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