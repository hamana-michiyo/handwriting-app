import 'package:flutter/material.dart';
import '../services/supabase_service.dart';

/// 手書きサンプル詳細画面
class SampleDetailScreen extends StatefulWidget {
  final Map<String, dynamic> sample;
  final List<Map<String, dynamic>>? allSamples;  // 全サンプルリスト（前後移動用）
  final int? currentIndex;  // 現在のインデックス（前後移動用）

  const SampleDetailScreen({
    super.key,
    required this.sample,
    this.allSamples,
    this.currentIndex,
  });

  @override
  State<SampleDetailScreen> createState() => _SampleDetailScreenState();
}

class _SampleDetailScreenState extends State<SampleDetailScreen> {
  final SupabaseService _supabaseService = SupabaseService();
  
  // 編集用コントローラー
  final Map<String, TextEditingController> _commentControllers = {};
  
  // UI状態
  bool _isEditing = true;  // デフォルトで編集モードをオンにする
  bool _isLoading = false;
  
  // データ
  late Map<String, dynamic> _currentSample;
  
  // 評価スコア（スライダー用）
  final Map<String, double> _scores = {
    'white': 0.0,
    'black': 0.0, 
    'center': 0.0,
    'shape': 0.0,
  };

  @override
  void initState() {
    super.initState();
    _currentSample = Map.from(widget.sample);
    print('サンプル初期化: $_currentSample');
    _initializeControllers();
  }

  @override
  void dispose() {
    _disposeControllers();
    super.dispose();
  }

  /// コントローラー初期化
  void _initializeControllers() {
    final scoreTypes = ['white', 'black', 'center', 'shape'];
    
    for (final type in scoreTypes) {
      // スコアを初期化
      final score = _currentSample['score_$type'] as int?;
      _scores[type] = score?.toDouble() ?? 0.0;
      
      // コメントコントローラーを初期化
      _commentControllers[type] = TextEditingController(
        text: _currentSample['comment_$type']?.toString() ?? '',
      );
    }
  }

  /// コントローラー破棄
  void _disposeControllers() {
    for (final controller in _commentControllers.values) {
      controller.dispose();
    }
  }

  /// 編集モード切り替え
  void _toggleEditMode() {
    setState(() {
      _isEditing = !_isEditing;
    });
  }

  /// 変更を保存
  Future<void> _saveChanges() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final sampleId = _currentSample['id'].toString();
      print('保存開始: サンプルID = $sampleId');
      
      // スコアと総合スコアを計算
      final scores = <String, int>{};
      double totalScore = 0;
      int validScores = 0;
      
      for (final type in ['white', 'black', 'center', 'shape']) {
        final score = _scores[type]?.round() ?? 0;
        if (score >= 0 && score <= 10) {
          scores['score_$type'] = score;
          totalScore += score;
          validScores++;
        }
      }
      
      // 総合スコア計算（重み付け平均）
      if (validScores > 0) {
        final weightedScore = (
          (scores['score_white'] ?? 0) * 0.3 +
          (scores['score_black'] ?? 0) * 0.2 +
          (scores['score_center'] ?? 0) * 0.2 +
          (scores['score_shape'] ?? 0) * 0.3
        );
        scores['score_overall'] = weightedScore.round();
      }
      
      // コメントを取得
      final comments = <String, String>{};
      for (final type in ['white', 'black', 'center', 'shape']) {
        final commentText = _commentControllers[type]?.text.trim() ?? '';
        if (commentText.isNotEmpty) {
          comments['comment_$type'] = commentText;
        }
      }
      
      print('保存データ: scores = $scores, comments = $comments');
      
      // Supabaseに保存
      final success = await _supabaseService.updateWritingSample(
        sampleId,
        scores: scores,
        comments: comments,
      );
      
      print('保存結果: success = $success');
      
      if (success) {
        // ローカルデータを更新
        setState(() {
          _currentSample.addAll(scores);
          _currentSample.addAll(comments);
          _currentSample['updated_at'] = DateTime.now().toIso8601String();
        });
        
        _showSuccessSnackBar('評価を保存しました');
      } else {
        _showErrorSnackBar('保存に失敗しました');
      }
    } catch (e) {
      _showErrorSnackBar('保存エラー: $e');
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  /// 変更をキャンセル
  void _cancelChanges() {
    // スコアとコメントを元の値にリセット
    final scoreTypes = ['white', 'black', 'center', 'shape'];
    
    for (final type in scoreTypes) {
      final score = _currentSample['score_$type'] as int?;
      _scores[type] = score?.toDouble() ?? 0.0;
      _commentControllers[type]?.text = _currentSample['comment_$type']?.toString() ?? '';
    }
    
    setState(() {
      _isEditing = false;
    });
  }

  /// 成功メッセージ表示
  void _showSuccessSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.green,
        duration: const Duration(seconds: 3),
      ),
    );
  }

  /// エラーメッセージ表示
  void _showErrorSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
        duration: const Duration(seconds: 5),
      ),
    );
  }

  /// 前のサンプルに移動
  void _goToPrevious() {
    if (widget.allSamples != null && widget.currentIndex != null) {
      final prevIndex = widget.currentIndex! - 1;
      if (prevIndex >= 0) {
        final prevSample = widget.allSamples![prevIndex];
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => SampleDetailScreen(
              sample: prevSample,
              allSamples: widget.allSamples,
              currentIndex: prevIndex,
            ),
          ),
        );
      }
    }
  }

  /// 次のサンプルに移動
  void _goToNext() {
    if (widget.allSamples != null && widget.currentIndex != null) {
      final nextIndex = widget.currentIndex! + 1;
      if (nextIndex < widget.allSamples!.length) {
        final nextSample = widget.allSamples![nextIndex];
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => SampleDetailScreen(
              sample: nextSample,
              allSamples: widget.allSamples,
              currentIndex: nextIndex,
            ),
          ),
        );
      }
    }
  }

  /// 前後移動可能か確認
  bool get _canGoToPrevious => widget.allSamples != null && 
      widget.currentIndex != null && widget.currentIndex! > 0;

  bool get _canGoToNext => widget.allSamples != null && 
      widget.currentIndex != null && widget.currentIndex! < widget.allSamples!.length - 1;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: _buildAppBar(),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildBasicInfoWithImage(),
                  const SizedBox(height: 20),
                  _buildEvaluationSection(),
                  const SizedBox(height: 20),
                  _buildNavigationButtons(),
                ],
              ),
            ),
    );
  }

  /// アプリバー
  PreferredSizeWidget _buildAppBar() {
    final currentPos = widget.currentIndex != null ? widget.currentIndex! + 1 : 0;
    final totalCount = widget.allSamples?.length ?? 0;
    final positionText = totalCount > 0 ? ' ($currentPos/$totalCount)' : '';
    
    return AppBar(
      title: Text('📄 サンプル詳細$positionText'),
      backgroundColor: Colors.blue.shade50,
      actions: [
        IconButton(
          icon: const Icon(Icons.save),
          onPressed: _saveChanges,
          tooltip: '保存',
        ),
        IconButton(
          icon: const Icon(Icons.refresh),
          onPressed: _cancelChanges,
          tooltip: 'リセット',
        ),
      ],
    );
  }

  /// 基本情報と画像を組み合わせたセクション
  Widget _buildBasicInfoWithImage() {
    // writer_numberを取得
    String writerNumber = '';
    try {
      final writerInfo = _currentSample['writers'] as Map<String, dynamic>?;
      if (writerInfo != null) {
        writerNumber = writerInfo['writer_number'] as String? ?? '';
      } else {
        final writerId = _currentSample['writer_id']?.toString() ?? '';
        writerNumber = 'writer_$writerId';
      }
    } catch (e) {
      final writerId = _currentSample['writer_id']?.toString() ?? '';
      writerNumber = 'writer_$writerId';
    }

    // 認識された文字を取得
    String recognizedChar = '';
    try {
      final geminiChar = _currentSample['gemini_recognized_char'] as String?;
      if (geminiChar != null && geminiChar.isNotEmpty) {
        recognizedChar = geminiChar;
      } else {
        final characterData = _currentSample['characters'] as Map<String, dynamic>?;
        if (characterData != null) {
          recognizedChar = characterData['character'] as String? ?? '';
        }
      }
    } catch (e) {
      recognizedChar = '不明';
    }

    // 学年・年代を取得
    String grade = '';
    try {
      final writerInfo = _currentSample['writers'] as Map<String, dynamic>?;
      if (writerInfo != null) {
        grade = writerInfo['grade'] as String? ?? '';
      }
    } catch (e) {
      grade = '';
    }

    final createdAt = DateTime.tryParse(_currentSample['created_at'] as String? ?? '') ?? DateTime.now();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 左側：小さな画像
            _buildSmallImage(),
            const SizedBox(width: 16),
            // 右側：基本情報
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '記入者No: ${writerNumber.replaceAll('writer_', '')}',
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '学年・年代: ${grade.isNotEmpty ? grade : '未設定'}',
                    style: const TextStyle(fontSize: 14),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '登録日: ${createdAt.toString().split(' ')[0]}',
                    style: const TextStyle(fontSize: 14),
                  ),
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.blue.shade100,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      '認識文字: $recognizedChar',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                        color: Colors.blue.shade800,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// 小さな画像ウィジェット
  Widget _buildSmallImage() {
    final imagePath = _currentSample['image_path'] as String?;
    
    if (imagePath == null || imagePath.isEmpty) {
      return Container(
        width: 80,
        height: 80,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade300),
          borderRadius: BorderRadius.circular(8),
          color: Colors.grey.shade100,
        ),
        child: Icon(
          Icons.image_not_supported,
          color: Colors.grey.shade400,
          size: 40,
        ),
      );
    }

    return Container(
      width: 80,
      height: 80,
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey.shade300),
        borderRadius: BorderRadius.circular(8),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: _buildNetworkImage(imagePath),
      ),
    );
  }




  /// ネットワーク画像ウィジェット（フォールバック機能付き）
  Widget _buildNetworkImage(String imagePath) {
    final publicUrl = _supabaseService.getImageUrl(imagePath);
    final authUrl = _supabaseService.getAuthenticatedImageUrl(imagePath);
    
    return Image.network(
      publicUrl,
      fit: BoxFit.contain,
      headers: const {
        'User-Agent': 'Flutter App',
      },
      loadingBuilder: (context, child, loadingProgress) {
        if (loadingProgress == null) return child;
        
        return Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(
                value: loadingProgress.expectedTotalBytes != null
                    ? loadingProgress.cumulativeBytesLoaded / loadingProgress.expectedTotalBytes!
                    : null,
              ),
              const SizedBox(height: 16),
              const Text(
                '画像を読み込み中...',
                style: TextStyle(color: Colors.grey),
              ),
            ],
          ),
        );
      },
      errorBuilder: (context, error, stackTrace) {
        debugPrint('公開URL画像読み込みエラー: $error');
        debugPrint('認証付きURLを試します: $authUrl');
        
        // 認証付きURLでリトライ
        return Image.network(
          authUrl,
          fit: BoxFit.contain,
          headers: {
            'Authorization': 'Bearer ${_supabaseService.supabaseAnonKey}',
            'apikey': _supabaseService.supabaseAnonKey,
            'User-Agent': 'Flutter App',
          },
          errorBuilder: (context, error2, stackTrace2) {
            debugPrint('認証付きURL画像読み込みエラー: $error2');
            return Container(
              color: Colors.grey.shade100,
              child: Icon(
                Icons.broken_image,
                color: Colors.grey.shade400,
                size: 30,
              ),
            );
          },
        );
      },
    );
  }



  /// 評価セクション
  Widget _buildEvaluationSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.star, color: Colors.orange),
                const SizedBox(width: 8),
                const Text(
                  '評価スコア',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const Spacer(),
                if (_currentSample['score_overall'] != null)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.orange.shade100,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: Colors.orange),
                    ),
                    child: Text(
                      '総合: ${_currentSample['score_overall']}点',
                      style: TextStyle(
                        color: Colors.orange.shade800,
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 16),
            _buildScoreSlider('白', 'white', '余白・バランス', Colors.blue),
            _buildScoreSlider('黒', 'black', '線の濃さ・太さ', Colors.black),
            _buildScoreSlider('場', 'center', '文字の配置・中心', Colors.red),
            _buildScoreSlider('形', 'shape', '文字の形・構造', Colors.green),
          ],
        ),
      ),
    );
  }


  /// スコアに応じた色を取得
  Color _getScoreColor(int? score) {
    if (score == null) return Colors.grey;
    if (score >= 8) return Colors.green;
    if (score >= 6) return Colors.orange;
    return Colors.red;
  }

  /// スコアスライダーウィジェット
  Widget _buildScoreSlider(String label, String type, String description, Color color) {
    final score = _scores[type] ?? 0.0;
    
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey.shade300),
        borderRadius: BorderRadius.circular(8),
        color: Colors.grey.shade50,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ラベル行
          Row(
            children: [
              Container(
                width: 24,
                height: 24,
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Center(
                  child: Text(
                    label,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  description,
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
              ),
              Text(
                '${score.round()}',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          // スライダー
          SliderTheme(
            data: SliderTheme.of(context).copyWith(
              activeTrackColor: color,
              inactiveTrackColor: color.withOpacity(0.3),
              thumbColor: color,
              thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 12),
              overlayColor: color.withOpacity(0.2),
              trackHeight: 8,
            ),
            child: Slider(
              value: score,
              min: 0,
              max: 10,
              divisions: 10,
              onChanged: (value) {
                setState(() {
                  _scores[type] = value;
                });
              },
            ),
          ),
          const SizedBox(height: 8),
          // コメント欄
          TextField(
            controller: _commentControllers[type],
            decoration: InputDecoration(
              hintText: '${label}のコメント',
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(8),
              ),
              filled: true,
              fillColor: Colors.white,
            ),
            maxLines: 2,
            style: const TextStyle(fontSize: 14),
          ),
        ],
      ),
    );
  }

  /// ナビゲーションボタン
  Widget _buildNavigationButtons() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        // 前へボタン
        SizedBox(
          width: 80,
          child: ElevatedButton(
            onPressed: _canGoToPrevious ? _goToPrevious : null,
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.blue,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 12),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8),
              ),
            ),
            child: const Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.arrow_back, size: 18),
                SizedBox(width: 4),
                Text('前へ', style: TextStyle(fontSize: 12)),
              ],
            ),
          ),
        ),
        // 更新ボタン
        SizedBox(
          width: 80,
          child: ElevatedButton(
            onPressed: _saveChanges,
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 12),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8),
              ),
            ),
            child: const Text('更新', style: TextStyle(fontSize: 12)),
          ),
        ),
        // 次へボタン
        SizedBox(
          width: 80,
          child: ElevatedButton(
            onPressed: _canGoToNext ? _goToNext : null,
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.blue,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 12),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8),
              ),
            ),
            child: const Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text('次へ', style: TextStyle(fontSize: 12)),
                SizedBox(width: 4),
                Icon(Icons.arrow_forward, size: 18),
              ],
            ),
          ),
        ),
      ],
    );
  }

}