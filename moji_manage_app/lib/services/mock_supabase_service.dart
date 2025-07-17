import 'dart:math';
import 'package:flutter/foundation.dart';
import '../models/capture_data.dart';

/// ローカルテスト用のモックSupabaseサービス
class MockSupabaseService {
  static const Duration _mockDelay = Duration(milliseconds: 500);
  
  // モックデータ
  static final List<Map<String, dynamic>> _mockData = [
    {
      'id': '1',
      'writer_number': '001',
      'created_at': '2024-01-15T10:30:00Z',
      'image_path': '/storage/sample1.jpg',
      'character_results': [
        {
          'gemini_result': {
            'character': '清',
            'confidence': 0.99,
          }
        }
      ],
      'score_white': 8,
      'score_black': 7,
      'score_center': 9,
      'score_shape': 8,
    },
    {
      'id': '2',
      'writer_number': '002',
      'created_at': '2024-01-15T11:15:00Z',
      'image_path': '/storage/sample2.jpg',
      'character_results': [
        {
          'gemini_result': {
            'character': '炎',
            'confidence': 0.98,
          }
        }
      ],
      'score_white': 7,
      'score_black': 8,
      'score_center': 6,
      'score_shape': 9,
    },
    {
      'id': '3',
      'writer_number': '003',
      'created_at': '2024-01-15T12:00:00Z',
      'image_path': '/storage/sample3.jpg',
      'character_results': [
        {
          'gemini_result': {
            'character': '葉',
            'confidence': 0.97,
          }
        }
      ],
      'score_white': 9,
      'score_black': 6,
      'score_center': 8,
      'score_shape': 7,
    },
    {
      'id': '4',
      'writer_number': '001',
      'created_at': '2024-01-15T13:30:00Z',
      'image_path': '/storage/sample4.jpg',
      'character_results': [
        {
          'gemini_result': {
            'character': '光',
            'confidence': 0.99,
          }
        }
      ],
      'score_white': 8,
      'score_black': 9,
      'score_center': 7,
      'score_shape': 8,
    },
    {
      'id': '5',
      'writer_number': '004',
      'created_at': '2024-01-15T14:15:00Z',
      'image_path': '/storage/sample5.jpg',
      'character_results': [
        {
          'gemini_result': {
            'character': '月',
            'confidence': 0.96,
          }
        }
      ],
      'score_white': null, // 未評価
      'score_black': null,
      'score_center': null,
      'score_shape': null,
    },
    {
      'id': '6',
      'writer_number': '005',
      'created_at': '2024-01-15T15:00:00Z',
      'image_path': '/storage/sample6.jpg',
      'character_results': [
        {
          'gemini_result': {
            'character': '花',
            'confidence': 0.94,
          }
        }
      ],
      'score_white': null, // 未評価
      'score_black': null,
      'score_center': null,
      'score_shape': null,
    },
    {
      'id': '7',
      'writer_number': '002',
      'created_at': '2024-01-15T16:30:00Z',
      'image_path': '/storage/sample7.jpg',
      'character_results': [
        {
          'gemini_result': {
            'character': '水',
            'confidence': 0.98,
          }
        }
      ],
      'score_white': 6,
      'score_black': 7,
      'score_center': 8,
      'score_shape': 6,
    },
    {
      'id': '8',
      'writer_number': '006',
      'created_at': '2024-01-15T17:15:00Z',
      'image_path': '/storage/sample8.jpg',
      'character_results': [
        {
          'gemini_result': {
            'character': '山',
            'confidence': 0.99,
          }
        }
      ],
      'score_white': 9,
      'score_black': 8,
      'score_center': 9,
      'score_shape': 9,
    },
    {
      'id': '9',
      'writer_number': '003',
      'created_at': '2024-01-15T18:00:00Z',
      'image_path': '/storage/sample9.jpg',
      'character_results': [
        {
          'gemini_result': {
            'character': '風',
            'confidence': 0.95,
          }
        }
      ],
      'score_white': null, // 未評価
      'score_black': null,
      'score_center': null,
      'score_shape': null,
    },
    {
      'id': '10',
      'writer_number': '007',
      'created_at': '2024-01-15T19:30:00Z',
      'image_path': '/storage/sample10.jpg',
      'character_results': [
        {
          'gemini_result': {
            'character': '海',
            'confidence': 0.97,
          }
        }
      ],
      'score_white': 7,
      'score_black': 8,
      'score_center': 7,
      'score_shape': 8,
    },
  ];

  /// 手書きサンプル一覧を取得
  Future<List<Map<String, dynamic>>> getWritingSamples({
    int limit = 100,
    int offset = 0,
    String orderBy = 'created_at',
    bool ascending = false,
  }) async {
    await Future.delayed(_mockDelay); // モック遅延
    
    debugPrint('Mock: サンプル取得開始 - limit: $limit, offset: $offset, orderBy: $orderBy');
    
    // ソート処理
    final sortedData = List<Map<String, dynamic>>.from(_mockData);
    sortedData.sort((a, b) {
      dynamic aVal = a[orderBy];
      dynamic bVal = b[orderBy];
      
      // null値の処理
      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return ascending ? -1 : 1;
      if (bVal == null) return ascending ? 1 : -1;
      
      // 日付の場合
      if (orderBy == 'created_at') {
        final aDate = DateTime.parse(aVal as String);
        final bDate = DateTime.parse(bVal as String);
        return ascending ? aDate.compareTo(bDate) : bDate.compareTo(aDate);
      }
      
      // その他の場合
      final comparison = aVal.toString().compareTo(bVal.toString());
      return ascending ? comparison : -comparison;
    });
    
    // ページネーション
    final start = offset;
    final end = (start + limit).clamp(0, sortedData.length);
    final result = sortedData.sublist(start, end);
    
    debugPrint('Mock: サンプル取得完了 - ${result.length}件');
    return result;
  }

  /// 記入者番号で検索
  Future<List<Map<String, dynamic>>> getWritingSamplesByWriter(String writerNumber) async {
    await Future.delayed(_mockDelay);
    
    debugPrint('Mock: 記入者検索開始 - $writerNumber');
    
    final result = _mockData
        .where((sample) => sample['writer_number'] == writerNumber)
        .toList();
    
    debugPrint('Mock: 記入者検索完了 - ${result.length}件');
    return result;
  }

  /// 文字で検索
  Future<List<Map<String, dynamic>>> getWritingSamplesByCharacter(String character) async {
    await Future.delayed(_mockDelay);
    
    debugPrint('Mock: 文字検索開始 - $character');
    
    final result = _mockData.where((sample) {
      try {
        final characterResults = sample['character_results'] as List<dynamic>?;
        if (characterResults != null) {
          for (final result in characterResults) {
            final resultMap = result as Map<String, dynamic>;
            final geminiResult = resultMap['gemini_result'] as Map<String, dynamic>?;
            if (geminiResult != null) {
              final sampleChar = geminiResult['character'] as String?;
              if (sampleChar == character) {
                return true;
              }
            }
          }
        }
      } catch (e) {
        debugPrint('Mock: 文字検索エラー - $e');
      }
      return false;
    }).toList();
    
    debugPrint('Mock: 文字検索完了 - ${result.length}件');
    return result;
  }

  /// 評価スコア範囲で検索
  Future<List<Map<String, dynamic>>> getWritingSamplesByScore({
    required String scoreType,
    required int minScore,
    required int maxScore,
  }) async {
    await Future.delayed(_mockDelay);
    
    debugPrint('Mock: スコア検索開始 - $scoreType($minScore-$maxScore)');
    
    final scoreColumn = 'score_$scoreType';
    final result = _mockData.where((sample) {
      final score = sample[scoreColumn] as int?;
      return score != null && score >= minScore && score <= maxScore;
    }).toList();
    
    debugPrint('Mock: スコア検索完了 - ${result.length}件');
    return result;
  }

  /// 統計情報を取得
  Future<Map<String, dynamic>> getStatistics() async {
    await Future.delayed(_mockDelay);
    
    debugPrint('Mock: 統計情報取得開始');
    
    final totalSamples = _mockData.length;
    final evaluatedSamples = _mockData.where((sample) => sample['score_white'] != null).length;
    final pendingSamples = totalSamples - evaluatedSamples;
    final uniqueWriters = _mockData.map((sample) => sample['writer_number']).toSet().length;
    
    final stats = {
      'total_samples': totalSamples,
      'evaluated_samples': evaluatedSamples,
      'pending_samples': pendingSamples,
      'total_writers': uniqueWriters,
      'unique_writers': uniqueWriters,
    };
    
    debugPrint('Mock: 統計情報取得完了 - $stats');
    return stats;
  }

  /// サンプル削除
  Future<bool> deleteWritingSample(String id) async {
    await Future.delayed(_mockDelay);
    
    debugPrint('Mock: サンプル削除開始 - $id');
    
    final index = _mockData.indexWhere((sample) => sample['id'] == id);
    if (index != -1) {
      _mockData.removeAt(index);
      debugPrint('Mock: サンプル削除完了 - $id');
      return true;
    }
    
    debugPrint('Mock: サンプル削除失敗 - $id (見つからない)');
    return false;
  }

  /// CaptureDataに変換
  CaptureData convertToCaptureData(Map<String, dynamic> supabaseData) {
    // 文字結果から文字ラベルを抽出
    final List<String> characterLabels = [];
    try {
      final characterResults = supabaseData['character_results'] as List<dynamic>?;
      if (characterResults != null) {
        for (final result in characterResults) {
          final resultMap = result as Map<String, dynamic>;
          final geminiResult = resultMap['gemini_result'] as Map<String, dynamic>?;
          if (geminiResult != null) {
            final character = geminiResult['character'] as String?;
            if (character != null && character.isNotEmpty) {
              characterLabels.add(character);
            }
          }
        }
      }
    } catch (e) {
      debugPrint('Mock: 文字ラベル抽出エラー - $e');
    }
    
    return CaptureData(
      imagePath: supabaseData['image_path'] as String?,
      writerNumber: supabaseData['writer_number'] as String? ?? '',
      captureTime: DateTime.tryParse(supabaseData['created_at'] as String? ?? '') ?? DateTime.now(),
      characterLabels: characterLabels,
      isProcessed: supabaseData['score_white'] != null,
    );
  }

  /// 追加のテストデータを生成
  void generateAdditionalTestData() {
    final random = Random();
    final characters = ['春', '夏', '秋', '冬', '雲', '雨', '雪', '空', '星', '虹'];
    final writers = ['008', '009', '010', '011', '012'];
    
    for (int i = 0; i < 20; i++) {
      final hasScore = random.nextBool();
      final writer = writers[random.nextInt(writers.length)];
      final character = characters[random.nextInt(characters.length)];
      
      _mockData.add({
        'id': '${_mockData.length + 1}',
        'writer_number': writer,
        'created_at': DateTime.now().subtract(Duration(hours: random.nextInt(72))).toIso8601String(),
        'image_path': '/storage/sample${_mockData.length + 1}.jpg',
        'character_results': [
          {
            'gemini_result': {
              'character': character,
              'confidence': 0.90 + random.nextDouble() * 0.09,
            }
          }
        ],
        'score_white': hasScore ? 6 + random.nextInt(4) : null,
        'score_black': hasScore ? 6 + random.nextInt(4) : null,
        'score_center': hasScore ? 6 + random.nextInt(4) : null,
        'score_shape': hasScore ? 6 + random.nextInt(4) : null,
      });
    }
    
    debugPrint('Mock: 追加テストデータ生成完了 - 総数: ${_mockData.length}');
  }
}