import 'package:flutter/material.dart';
import '../services/supabase_service.dart';
import '../services/mock_supabase_service.dart';
import 'sample_detail_screen.dart';

/// 手書きサンプル一覧管理画面
class SampleListScreen extends StatefulWidget {
  const SampleListScreen({super.key});

  @override
  State<SampleListScreen> createState() => _SampleListScreenState();
}

class _SampleListScreenState extends State<SampleListScreen> {
  // デバッグ用フラグ - 実際のSupabaseを使用する場合はfalseに設定
  static const bool _useMockMode = false;
  
  final SupabaseService _supabaseService = SupabaseService();
  final MockSupabaseService _mockService = MockSupabaseService();
  final TextEditingController _searchController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  
  // データ管理
  List<Map<String, dynamic>> _allSamples = [];
  List<Map<String, dynamic>> _filteredSamples = [];
  Map<String, dynamic> _statistics = {};
  
  // UI状態
  bool _isLoading = false;
  bool _isLoadingMore = false;
  String _searchQuery = '';
  String _selectedFilter = 'all'; // all, evaluated, pending
  String _selectedSortBy = 'created_at'; // created_at, writer_number, score_white
  bool _isAscending = false;
  
  // ページネーション
  int _currentPage = 0;
  static const int _pageSize = 50;
  bool _hasMoreData = true;

  @override
  void initState() {
    super.initState();
    // モックサービスの場合は追加テストデータを生成
    if (_useMockMode) {
      _mockService.generateAdditionalTestData();
    }
    _loadInitialData();
    _scrollController.addListener(_onScroll);
  }

  @override
  void dispose() {
    _searchController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  /// 初期データ読み込み
  Future<void> _loadInitialData() async {
    setState(() {
      _isLoading = true;
    });

    try {
      // 統計情報を取得
      final stats = _useMockMode 
          ? await _mockService.getStatistics()
          : await _supabaseService.getStatistics();
      
      // 最初のページを取得
      final samples = _useMockMode
          ? await _mockService.getWritingSamples(
              limit: _pageSize,
              offset: 0,
              orderBy: _selectedSortBy,
              ascending: _isAscending,
            )
          : await _supabaseService.getWritingSamples(
              limit: _pageSize,
              offset: 0,
              orderBy: _selectedSortBy,
              ascending: _isAscending,
            );

      if (mounted) {
        setState(() {
          _statistics = stats;
          _allSamples = samples;
          _filteredSamples = samples;
          _currentPage = 0;
          _hasMoreData = samples.length == _pageSize;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        _showErrorSnackBar('データの読み込みに失敗しました: $e');
      }
    }
  }

  /// 追加データ読み込み（ページネーション）
  Future<void> _loadMoreData() async {
    if (_isLoadingMore || !_hasMoreData) return;

    setState(() {
      _isLoadingMore = true;
    });

    try {
      final newSamples = _useMockMode
          ? await _mockService.getWritingSamples(
              limit: _pageSize,
              offset: (_currentPage + 1) * _pageSize,
              orderBy: _selectedSortBy,
              ascending: _isAscending,
            )
          : await _supabaseService.getWritingSamples(
              limit: _pageSize,
              offset: (_currentPage + 1) * _pageSize,
              orderBy: _selectedSortBy,
              ascending: _isAscending,
            );

      if (mounted) {
        setState(() {
          _allSamples.addAll(newSamples);
          _applyFilters();
          _currentPage++;
          _hasMoreData = newSamples.length == _pageSize;
          _isLoadingMore = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoadingMore = false;
        });
        _showErrorSnackBar('追加データの読み込みに失敗しました: $e');
      }
    }
  }

  /// スクロール監視
  void _onScroll() {
    if (_scrollController.position.pixels >= _scrollController.position.maxScrollExtent - 200) {
      _loadMoreData();
    }
  }

  /// フィルタ適用
  void _applyFilters() {
    List<Map<String, dynamic>> filtered = List.from(_allSamples);

    // 検索クエリでフィルタリング
    if (_searchQuery.isNotEmpty) {
      filtered = filtered.where((sample) {
        final query = _searchQuery.toLowerCase();
        
        // 記入者番号での検索（新しいスキーマ対応）
        try {
          final writerInfo = sample['writers'] as Map<String, dynamic>?;
          if (writerInfo != null) {
            final writerNumber = writerInfo['writer_number']?.toString().toLowerCase() ?? '';
            if (writerNumber.contains(query)) return true;
          }
        } catch (e) {
          debugPrint('記入者番号検索エラー: $e');
        }
        
        // 認識された文字での検索
        try {
          // Gemini認識文字での検索
          final geminiChar = sample['gemini_recognized_char'] as String?;
          if (geminiChar != null && geminiChar.toLowerCase().contains(query)) {
            return true;
          }
          
          // charactersテーブルからの文字情報での検索
          final characterInfo = sample['characters'] as Map<String, dynamic>?;
          if (characterInfo != null) {
            final character = characterInfo['character'] as String?;
            if (character != null && character.toLowerCase().contains(query)) {
              return true;
            }
          }
        } catch (e) {
          debugPrint('文字検索エラー: $e');
        }
        
        return false;
      }).toList();
    }

    // 評価状態でフィルタリング
    if (_selectedFilter == 'evaluated') {
      filtered = filtered.where((sample) => sample['score_white'] != null).toList();
    } else if (_selectedFilter == 'pending') {
      filtered = filtered.where((sample) => sample['score_white'] == null).toList();
    }

    setState(() {
      _filteredSamples = filtered;
    });
  }

  /// 検索実行
  void _performSearch(String query) {
    setState(() {
      _searchQuery = query;
    });
    _applyFilters();
  }

  /// フィルタ変更
  void _changeFilter(String filter) {
    setState(() {
      _selectedFilter = filter;
    });
    _applyFilters();
  }

  /// ソート変更
  void _changeSort(String sortBy) {
    setState(() {
      _selectedSortBy = sortBy;
      _isAscending = !_isAscending;
    });
    _loadInitialData(); // データを再取得
  }

  /// サンプル削除
  Future<void> _deleteSample(String id) async {
    final confirmed = await _showDeleteConfirmDialog();
    if (!confirmed) return;

    try {
      final success = _useMockMode
          ? await _mockService.deleteWritingSample(id)
          : await _supabaseService.deleteWritingSample(id);
      if (success) {
        setState(() {
          _allSamples.removeWhere((sample) => sample['id'] == id);
          _applyFilters();
        });
        _showSuccessSnackBar('サンプルを削除しました');
      } else {
        _showErrorSnackBar('削除に失敗しました');
      }
    } catch (e) {
      _showErrorSnackBar('削除エラー: $e');
    }
  }

  /// 削除確認ダイアログ
  Future<bool> _showDeleteConfirmDialog() async {
    return await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('⚠️ 削除確認'),
        content: const Text('このサンプルを削除しますか？\n削除した場合は元に戻せません。'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('キャンセル'),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('削除'),
          ),
        ],
      ),
    ) ?? false;
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('📋 手書きサンプル一覧'),
        backgroundColor: Colors.green.shade50,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadInitialData,
            tooltip: 'データを更新',
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                _buildStatisticsHeader(),
                _buildSearchAndFilters(),
                _buildSampleList(),
              ],
            ),
    );
  }

  /// 統計情報ヘッダー
  Widget _buildStatisticsHeader() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.green.shade50,
        border: Border(bottom: BorderSide(color: Colors.grey.shade300)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildStatChip('総数', '${_statistics['total_samples'] ?? 0}', Colors.blue),
          _buildStatChip('評価済み', '${_statistics['evaluated_samples'] ?? 0}', Colors.green),
          _buildStatChip('未評価', '${_statistics['pending_samples'] ?? 0}', Colors.orange),
          _buildStatChip('記入者', '${_statistics['total_writers'] ?? 0}人', Colors.purple),
        ],
      ),
    );
  }

  /// 統計チップ
  Widget _buildStatChip(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          value,
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            color: Colors.grey.shade600,
          ),
        ),
      ],
    );
  }

  /// 検索・フィルタ部分
  Widget _buildSearchAndFilters() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border(bottom: BorderSide(color: Colors.grey.shade300)),
      ),
      child: Column(
        children: [
          // 検索バー
          TextField(
            controller: _searchController,
            decoration: InputDecoration(
              hintText: '記入者番号または文字で検索...',
              prefixIcon: const Icon(Icons.search),
              suffixIcon: _searchQuery.isNotEmpty
                  ? IconButton(
                      icon: const Icon(Icons.clear),
                      onPressed: () {
                        _searchController.clear();
                        _performSearch('');
                      },
                    )
                  : null,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(8),
              ),
            ),
            onChanged: _performSearch,
          ),
          const SizedBox(height: 12),
          // フィルタ・ソートボタン
          Row(
            children: [
              // フィルタ選択
              Expanded(
                child: DropdownButtonFormField<String>(
                  value: _selectedFilter,
                  decoration: const InputDecoration(
                    labelText: 'フィルタ',
                    border: OutlineInputBorder(),
                  ),
                  items: const [
                    DropdownMenuItem(value: 'all', child: Text('すべて')),
                    DropdownMenuItem(value: 'evaluated', child: Text('評価済み')),
                    DropdownMenuItem(value: 'pending', child: Text('未評価')),
                  ],
                  onChanged: (value) {
                    if (value != null) _changeFilter(value);
                  },
                ),
              ),
              const SizedBox(width: 12),
              // ソート選択
              Expanded(
                child: DropdownButtonFormField<String>(
                  value: _selectedSortBy,
                  decoration: const InputDecoration(
                    labelText: 'ソート',
                    border: OutlineInputBorder(),
                  ),
                  items: const [
                    DropdownMenuItem(value: 'created_at', child: Text('作成日時')),
                    DropdownMenuItem(value: 'writer_number', child: Text('記入者番号')),
                    DropdownMenuItem(value: 'score_white', child: Text('白スコア')),
                  ],
                  onChanged: (value) {
                    if (value != null) _changeSort(value);
                  },
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  /// サンプル一覧
  Widget _buildSampleList() {
    return Expanded(
      child: _filteredSamples.isEmpty
          ? const Center(
              child: Text(
                '該当するサンプルがありません',
                style: TextStyle(color: Colors.grey),
              ),
            )
          : ListView.builder(
              controller: _scrollController,
              itemCount: _filteredSamples.length + (_isLoadingMore ? 1 : 0),
              itemBuilder: (context, index) {
                if (index == _filteredSamples.length) {
                  return const Center(
                    child: Padding(
                      padding: EdgeInsets.all(16),
                      child: CircularProgressIndicator(),
                    ),
                  );
                }

                final sample = _filteredSamples[index];
                return _buildSampleCard(sample);
              },
            ),
    );
  }

  /// サンプルカード
  Widget _buildSampleCard(Map<String, dynamic> sample) {
    // 現在のサンプルのインデックスを取得
    final currentIndex = _filteredSamples.indexOf(sample);
    // 基本データから情報を取得
    final writerId = sample['writer_id']?.toString() ?? '';
    final createdAt = DateTime.tryParse(sample['created_at'] as String? ?? '') ?? DateTime.now();
    final isEvaluated = sample['score_white'] != null;
    
    // writer_numberを取得（JOINデータがある場合）
    String writerNumber = 'writer_$writerId';
    try {
      final writerInfo = sample['writers'] as Map<String, dynamic>?;
      if (writerInfo != null) {
        writerNumber = writerInfo['writer_number'] as String? ?? 'writer_$writerId';
      }
    } catch (e) {
      debugPrint('記入者番号抽出エラー: $e');
    }
    
    // 認識された文字を取得
    final recognizedCharacters = <String>[];
    try {
      // Gemini認識文字を追加
      final geminiChar = sample['gemini_recognized_char'] as String?;
      if (geminiChar != null && geminiChar.isNotEmpty) {
        recognizedCharacters.add(geminiChar);
      }
      
      // charactersテーブルからの文字情報も追加（JOINデータがある場合）
      final characterInfo = sample['characters'] as Map<String, dynamic>?;
      if (characterInfo != null) {
        final character = characterInfo['character'] as String?;
        if (character != null && character.isNotEmpty && !recognizedCharacters.contains(character)) {
          recognizedCharacters.add(character);
        }
      }
      
      // character_idから推測（基本データのみの場合）
      if (recognizedCharacters.isEmpty) {
        final characterId = sample['character_id']?.toString();
        if (characterId != null) {
          recognizedCharacters.add('文字ID:$characterId');
        }
      }
    } catch (e) {
      debugPrint('文字抽出エラー: $e');
    }

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: InkWell(
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => SampleDetailScreen(
                sample: sample,
                allSamples: _filteredSamples,
                currentIndex: currentIndex,
              ),
            ),
          );
        },
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ヘッダー行
              Row(
                children: [
                  // 画像プレビュー
                  _buildImagePreview(sample),
                  const SizedBox(width: 12),
                  // メイン情報
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '記入者: $writerNumber',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                              decoration: BoxDecoration(
                                color: isEvaluated ? Colors.green : Colors.orange,
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: Text(
                                isEvaluated ? '評価済み' : '未評価',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 12,
                                ),
                              ),
                            ),
                            const Spacer(),
                            PopupMenuButton(
                              icon: const Icon(Icons.more_vert),
                              itemBuilder: (context) => [
                                const PopupMenuItem(
                                  value: 'details',
                                  child: Row(
                                    children: [
                                      Icon(Icons.info_outline),
                                      SizedBox(width: 8),
                                      Text('詳細表示'),
                                    ],
                                  ),
                                ),
                                PopupMenuItem(
                                  value: 'delete',
                                  child: const Row(
                                    children: [
                                      Icon(Icons.delete_outline, color: Colors.red),
                                      SizedBox(width: 8),
                                      Text('削除', style: TextStyle(color: Colors.red)),
                                    ],
                                  ),
                                ),
                              ],
                              onSelected: (value) {
                                if (value == 'delete') {
                                  _deleteSample(sample['id'] as String);
                                } else if (value == 'details') {
                                  Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                      builder: (context) => SampleDetailScreen(
                                        sample: sample,
                                        allSamples: _filteredSamples,
                                        currentIndex: currentIndex,
                                      ),
                                    ),
                                  );
                                }
                              },
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              // 認識された文字
              if (recognizedCharacters.isNotEmpty)
                Text(
                  '認識文字: ${recognizedCharacters.join(', ')}',
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey.shade700,
                  ),
                ),
              const SizedBox(height: 8),
              // 評価スコア
              if (isEvaluated)
                Row(
                  children: [
                    _buildScoreChip('白', sample['score_white']),
                    const SizedBox(width: 8),
                    _buildScoreChip('黒', sample['score_black']),
                    const SizedBox(width: 8),
                    _buildScoreChip('場', sample['score_center']),
                    const SizedBox(width: 8),
                    _buildScoreChip('形', sample['score_shape']),
                  ],
                ),
              const SizedBox(height: 8),
              // 作成日時
              Text(
                '作成: ${createdAt.toString().split('.')[0]}',
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey.shade600,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// 評価スコアチップ
  Widget _buildScoreChip(String label, dynamic score) {
    final scoreValue = score as int? ?? 0;
    final color = scoreValue >= 8 ? Colors.green : scoreValue >= 6 ? Colors.orange : Colors.red;
    
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        border: Border.all(color: color),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        '$label:$scoreValue',
        style: TextStyle(
          fontSize: 12,
          color: color,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  /// 画像プレビューウィジェット
  Widget _buildImagePreview(Map<String, dynamic> sample) {
    final imagePath = sample['image_path'] as String?;
    
    if (imagePath == null || imagePath.isEmpty) {
      return Container(
        width: 60,
        height: 60,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade300),
          borderRadius: BorderRadius.circular(8),
          color: Colors.grey.shade100,
        ),
        child: Icon(
          Icons.image_not_supported,
          color: Colors.grey.shade400,
          size: 30,
        ),
      );
    }

    final imageUrl = _useMockMode 
        ? '' // モックモードでは画像なし
        : _supabaseService.getImageUrl(imagePath);
    
    if (imageUrl.isEmpty) {
      return Container(
        width: 60,
        height: 60,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade300),
          borderRadius: BorderRadius.circular(8),
          color: Colors.grey.shade100,
        ),
        child: Icon(
          Icons.image,
          color: Colors.grey.shade400,
          size: 30,
        ),
      );
    }

    return Container(
      width: 60,
      height: 60,
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey.shade300),
        borderRadius: BorderRadius.circular(8),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: Image.network(
          imageUrl,
          fit: BoxFit.cover,
          loadingBuilder: (context, child, loadingProgress) {
            if (loadingProgress == null) return child;
            
            return Container(
              color: Colors.grey.shade100,
              child: Center(
                child: SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    value: loadingProgress.expectedTotalBytes != null
                        ? loadingProgress.cumulativeBytesLoaded / loadingProgress.expectedTotalBytes!
                        : null,
                  ),
                ),
              ),
            );
          },
          errorBuilder: (context, error, stackTrace) {
            return Container(
              color: Colors.grey.shade100,
              child: Icon(
                Icons.broken_image,
                color: Colors.grey.shade400,
                size: 30,
              ),
            );
          },
        ),
      ),
    );
  }
}