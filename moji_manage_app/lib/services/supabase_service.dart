import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import '../models/capture_data.dart';

/// Supabaseデータベースとの通信を担当するサービスクラス
class SupabaseService {
  // Supabase設定 - 実際のプロジェクト認証情報
  static const String _supabaseUrl = 'https://ypobmpkecniyuawxukol.supabase.co';
  static const String _supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlwb2JtcGtlY25peXVhd3h1a29sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE3ODEzNzMsImV4cCI6MjA2NzM1NzM3M30.JdrURiuZJ4HvFo32bUTfr3ELLRS8BzFhBBldapvzGjw';
  static const Duration _timeout = Duration(seconds: 30);
  
  // 外部からアクセス可能なゲッター
  String get supabaseAnonKey => _supabaseAnonKey;
  
  // テーブル名
  static const String _writingSampleTable = 'writing_samples';
  
  // Supabase Storage設定
  static const String _storageBucket = 'ml-data';
  
  /// 認証ヘッダーを取得
  Map<String, String> get _headers => {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer $_supabaseAnonKey',
    'apikey': _supabaseAnonKey,
    'Prefer': 'return=representation',
  };
  
  /// 手書きサンプル一覧を取得
  /// 
  /// [limit] - 取得する最大件数（デフォルト: 100）
  /// [offset] - 取得開始位置（デフォルト: 0）
  /// [orderBy] - ソート列（デフォルト: created_at）
  /// [ascending] - 昇順ソート（デフォルト: false）
  /// 
  /// Returns: 手書きサンプルのリスト
  Future<List<Map<String, dynamic>>> getWritingSamples({
    int limit = 100,
    int offset = 0,
    String orderBy = 'created_at',
    bool ascending = false,
  }) async {
    try {
      // まず基本的なクエリでテスト
      final basicUri = Uri.parse('$_supabaseUrl/rest/v1/$_writingSampleTable')
          .replace(queryParameters: {
        'select': '*',
        'limit': limit.toString(),
        'offset': offset.toString(),
        'order': '$orderBy.${ascending ? 'asc' : 'desc'}',
      });
      
      debugPrint('Supabase基本取得開始: $basicUri');
      
      final basicResponse = await http.get(basicUri, headers: _headers).timeout(_timeout);
      
      if (basicResponse.statusCode == 200) {
        final List<dynamic> basicData = jsonDecode(basicResponse.body);
        debugPrint('Supabase基本取得成功: ${basicData.length}件');
        debugPrint('基本データ例: ${basicData.isNotEmpty ? basicData[0] : 'なし'}');
        
        // 基本データが取得できた場合、JOINを試す
        if (basicData.isNotEmpty) {
          final uri = Uri.parse('$_supabaseUrl/rest/v1/$_writingSampleTable')
              .replace(queryParameters: {
            'select': '*,writers(writer_number,age,grade),characters(character,stroke_count,difficulty_level,category)',
            'limit': limit.toString(),
            'offset': offset.toString(),
            'order': '$orderBy.${ascending ? 'asc' : 'desc'}',
          });
          
          debugPrint('Supabase JOIN取得開始: $uri');
          
          final response = await http.get(uri, headers: _headers).timeout(_timeout);
          
          if (response.statusCode == 200) {
            final List<dynamic> data = jsonDecode(response.body);
            debugPrint('Supabase JOIN取得成功: ${data.length}件');
            return data.cast<Map<String, dynamic>>();
          } else {
            debugPrint('Supabase JOIN取得エラー: ${response.statusCode}');
            debugPrint('エラー内容: ${response.body}');
            // JOINが失敗した場合、基本データを返す
            return basicData.cast<Map<String, dynamic>>();
          }
        }
        
        return basicData.cast<Map<String, dynamic>>();
      } else {
        debugPrint('Supabase基本取得エラー: ${basicResponse.statusCode}');
        debugPrint('エラー内容: ${basicResponse.body}');
        return [];
      }
    } catch (e) {
      debugPrint('Supabase取得失敗: $e');
      return [];
    }
  }
  
  /// 記入者番号で手書きサンプルを検索
  /// 
  /// [writerNumber] - 記入者番号
  /// 
  /// Returns: 該当する手書きサンプルのリスト
  Future<List<Map<String, dynamic>>> getWritingSamplesByWriter(String writerNumber) async {
    try {
      final uri = Uri.parse('$_supabaseUrl/rest/v1/$_writingSampleTable')
          .replace(queryParameters: {
        'select': '*',
        'writer_number': 'eq.$writerNumber',
        'order': 'created_at.desc',
      });
      
      debugPrint('記入者検索開始: $writerNumber');
      
      final response = await http.get(uri, headers: _headers).timeout(_timeout);
      
      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        debugPrint('記入者検索成功: ${data.length}件');
        return data.cast<Map<String, dynamic>>();
      } else {
        debugPrint('記入者検索エラー: ${response.statusCode}');
        return [];
      }
    } catch (e) {
      debugPrint('記入者検索失敗: $e');
      return [];
    }
  }
  
  /// 文字で手書きサンプルを検索
  /// 
  /// [character] - 検索する文字
  /// 
  /// Returns: 該当する手書きサンプルのリスト
  Future<List<Map<String, dynamic>>> getWritingSamplesByCharacter(String character) async {
    try {
      final uri = Uri.parse('$_supabaseUrl/rest/v1/$_writingSampleTable')
          .replace(queryParameters: {
        'select': '*',
        'character_results': 'cs.{"gemini_result":{"character":"$character"}}',
        'order': 'created_at.desc',
      });
      
      debugPrint('文字検索開始: $character');
      
      final response = await http.get(uri, headers: _headers).timeout(_timeout);
      
      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        debugPrint('文字検索成功: ${data.length}件');
        return data.cast<Map<String, dynamic>>();
      } else {
        debugPrint('文字検索エラー: ${response.statusCode}');
        return [];
      }
    } catch (e) {
      debugPrint('文字検索失敗: $e');
      return [];
    }
  }
  
  /// 評価スコア範囲で手書きサンプルを検索
  /// 
  /// [scoreType] - スコア種別（'white', 'black', 'center', 'shape'）
  /// [minScore] - 最小スコア
  /// [maxScore] - 最大スコア
  /// 
  /// Returns: 該当する手書きサンプルのリスト
  Future<List<Map<String, dynamic>>> getWritingSamplesByScore({
    required String scoreType,
    required int minScore,
    required int maxScore,
  }) async {
    try {
      final scoreColumn = 'score_$scoreType';
      final uri = Uri.parse('$_supabaseUrl/rest/v1/$_writingSampleTable')
          .replace(queryParameters: {
        'select': '*',
        '$scoreColumn': 'gte.$minScore',
        '$scoreColumn': 'lte.$maxScore',
        'order': '$scoreColumn.desc',
      });
      
      debugPrint('スコア検索開始: $scoreType($minScore-$maxScore)');
      
      final response = await http.get(uri, headers: _headers).timeout(_timeout);
      
      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        debugPrint('スコア検索成功: ${data.length}件');
        return data.cast<Map<String, dynamic>>();
      } else {
        debugPrint('スコア検索エラー: ${response.statusCode}');
        return [];
      }
    } catch (e) {
      debugPrint('スコア検索失敗: $e');
      return [];
    }
  }
  
  /// 統計情報を取得
  /// 
  /// Returns: 統計情報のマップ
  Future<Map<String, dynamic>> getStatistics() async {
    try {
      // まず基本的なクエリでテスト（quality_statusを含む）
      final basicUri = Uri.parse('$_supabaseUrl/rest/v1/$_writingSampleTable')
          .replace(queryParameters: {
        'select': 'score_white,writer_id,quality_status',
      });
      
      debugPrint('統計情報基本取得開始: $basicUri');
      
      final basicResponse = await http.get(basicUri, headers: _headers).timeout(_timeout);
      
      if (basicResponse.statusCode == 200) {
        final List<dynamic> basicData = jsonDecode(basicResponse.body);
        debugPrint('統計情報基本取得成功: ${basicData.length}件');
        debugPrint('基本統計データ例: ${basicData.isNotEmpty ? basicData[0] : 'なし'}');
        
        // 0件の場合、RLSポリシーの影響を確認
        if (basicData.isEmpty) {
          debugPrint('⚠️ データが0件です。RLSポリシーによる制限の可能性があります。');
          debugPrint('現在のポリシー: quality_status = "approved" のデータのみ閲覧可能');
          
          // 全データアクセステスト（quality_statusフィルタなし）
          final allDataUri = Uri.parse('$_supabaseUrl/rest/v1/$_writingSampleTable')
              .replace(queryParameters: {
            'select': 'id,quality_status',
            'limit': '5', // 最初の5件のみ
          });
          
          debugPrint('全データアクセステスト: $allDataUri');
          
          final allDataResponse = await http.get(allDataUri, headers: _headers).timeout(_timeout);
          
          if (allDataResponse.statusCode == 200) {
            final List<dynamic> allData = jsonDecode(allDataResponse.body);
            debugPrint('全データアクセス結果: ${allData.length}件');
            debugPrint('全データ例: ${allData.isNotEmpty ? allData[0] : 'なし'}');
          } else {
            debugPrint('全データアクセスエラー: ${allDataResponse.statusCode}');
            debugPrint('エラー内容: ${allDataResponse.body}');
          }
        }
        
        final totalCount = basicData.length;
        final evaluatedCount = basicData.where((item) => item['score_white'] != null).length;
        final uniqueWriters = basicData.map((item) => item['writer_id']).where((w) => w != null).toSet().length;
        
        debugPrint('統計情報取得成功: 総数=$totalCount, 評価済み=$evaluatedCount, 記入者数=$uniqueWriters');
        
        return {
          'total_samples': totalCount,
          'evaluated_samples': evaluatedCount,
          'pending_samples': totalCount - evaluatedCount,
          'total_writers': uniqueWriters,
          'unique_writers': uniqueWriters,
        };
      } else {
        debugPrint('統計情報基本取得エラー: ${basicResponse.statusCode}');
        debugPrint('エラー内容: ${basicResponse.body}');
        return {
          'total_samples': 0,
          'evaluated_samples': 0,
          'pending_samples': 0,
          'total_writers': 0,
          'unique_writers': 0,
        };
      }
    } catch (e) {
      debugPrint('統計情報取得失敗: $e');
      return {
        'total_samples': 0,
        'evaluated_samples': 0,
        'pending_samples': 0,
        'total_writers': 0,
        'unique_writers': 0,
      };
    }
  }
  
  /// 手書きサンプルを削除
  /// 
  /// [id] - 削除するサンプルのID
  /// 
  /// Returns: 削除成功時はtrue
  Future<bool> deleteWritingSample(String id) async {
    try {
      final uri = Uri.parse('$_supabaseUrl/rest/v1/$_writingSampleTable')
          .replace(queryParameters: {
        'id': 'eq.$id',
      });
      
      debugPrint('サンプル削除開始: $id');
      
      final response = await http.delete(uri, headers: _headers).timeout(_timeout);
      
      if (response.statusCode == 204) {
        debugPrint('サンプル削除成功: $id');
        return true;
      } else {
        debugPrint('サンプル削除エラー: ${response.statusCode}');
        return false;
      }
    } catch (e) {
      debugPrint('サンプル削除失敗: $e');
      return false;
    }
  }

  /// 手書きサンプルを更新
  /// 
  /// [id] - 更新するサンプルのID
  /// [scores] - 更新するスコア（score_white, score_black等）
  /// [comments] - 更新するコメント（comment_white, comment_black等）
  /// 
  /// Returns: 更新成功時はtrue
  Future<bool> updateWritingSample(
    String id, {
    Map<String, dynamic>? scores,
    Map<String, String>? comments,
  }) async {
    try {
      if (scores == null && comments == null) {
        debugPrint('更新データがありません');
        return false;
      }

      final updateData = <String, dynamic>{};
      
      // スコアを追加
      if (scores != null) {
        updateData.addAll(scores);
      }
      
      // コメントを追加
      if (comments != null) {
        updateData.addAll(comments);
      }
      
      // 更新日時を設定
      updateData['updated_at'] = DateTime.now().toIso8601String();
      
      final uri = Uri.parse('$_supabaseUrl/rest/v1/$_writingSampleTable')
          .replace(queryParameters: {
        'id': 'eq.$id',
      });
      
      debugPrint('サンプル更新開始: $id');
      debugPrint('更新データ: $updateData');
      
      final response = await http.patch(
        uri,
        headers: _headers,
        body: jsonEncode(updateData),
      ).timeout(_timeout);
      
      if (response.statusCode == 200 || response.statusCode == 204) {
        debugPrint('サンプル更新成功: $id');
        if (response.statusCode == 200) {
          // 更新されたデータが返された場合、ログに出力
          debugPrint('更新されたデータ: ${response.body}');
        }
        return true;
      } else {
        debugPrint('サンプル更新エラー: ${response.statusCode}');
        debugPrint('エラー内容: ${response.body}');
        return false;
      }
    } catch (e) {
      debugPrint('サンプル更新失敗: $e');
      return false;
    }
  }

  /// Supabase Storageから画像URLを取得
  /// 
  /// [imagePath] - Storageでの画像パス
  /// 
  /// Returns: 画像の公開URL
  String getImageUrl(String imagePath) {
    try {
      // image_pathがフルパスの場合、相対パスに変換
      String relativePath = imagePath;
      if (imagePath.startsWith('/')) {
        relativePath = imagePath.substring(1);
      }
      
      // パスの各部分を個別にエンコード
      final pathParts = relativePath.split('/');
      final encodedParts = pathParts.map((part) => Uri.encodeComponent(part)).toList();
      final encodedPath = encodedParts.join('/');
      
      // Supabase Storage の公開URL形式
      final imageUrl = '$_supabaseUrl/storage/v1/object/public/$_storageBucket/$encodedPath';
      
      debugPrint('画像URL生成: $imagePath → $imageUrl');
      return imageUrl;
    } catch (e) {
      debugPrint('画像URL生成エラー: $e');
      return '';
    }
  }

  /// 認証付きSupabase Storage画像URLを取得
  /// 
  /// [imagePath] - Storageでの画像パス
  /// 
  /// Returns: 認証付き画像URL
  String getAuthenticatedImageUrl(String imagePath) {
    try {
      // image_pathがフルパスの場合、相対パスに変換
      String relativePath = imagePath;
      if (imagePath.startsWith('/')) {
        relativePath = imagePath.substring(1);
      }
      
      // URLエンコーディングを適用
      final encodedPath = Uri.encodeFull(relativePath);
      
      // 認証付きSupabase Storage API形式
      final imageUrl = '$_supabaseUrl/storage/v1/object/$_storageBucket/$encodedPath';
      
      debugPrint('認証付き画像URL生成: $imagePath → $imageUrl');
      return imageUrl;
    } catch (e) {
      debugPrint('認証付き画像URL生成エラー: $e');
      return '';
    }
  }

  /// 画像の存在確認（複数方式でテスト）
  /// 
  /// [imagePath] - 確認する画像パス
  /// 
  /// Returns: 画像が存在する場合はtrue
  Future<bool> checkImageExists(String imagePath) async {
    try {
      debugPrint('画像存在確認開始: $imagePath');
      
      // 方式1: 公開URL
      final publicUrl = getImageUrl(imagePath);
      if (publicUrl.isNotEmpty) {
        try {
          final publicResponse = await http.head(Uri.parse(publicUrl)).timeout(
            const Duration(seconds: 10),
          );
          if (publicResponse.statusCode == 200) {
            debugPrint('画像存在確認成功（公開URL）: $imagePath');
            return true;
          }
          debugPrint('公開URL失敗: ${publicResponse.statusCode}');
        } catch (e) {
          debugPrint('公開URLエラー: $e');
        }
      }
      
      // 方式2: 認証付きURL
      final authUrl = getAuthenticatedImageUrl(imagePath);
      if (authUrl.isNotEmpty) {
        try {
          final authResponse = await http.head(
            Uri.parse(authUrl),
            headers: _headers,
          ).timeout(const Duration(seconds: 10));
          
          if (authResponse.statusCode == 200) {
            debugPrint('画像存在確認成功（認証URL）: $imagePath');
            return true;
          }
          debugPrint('認証URL失敗: ${authResponse.statusCode}');
        } catch (e) {
          debugPrint('認証URLエラー: $e');
        }
      }
      
      debugPrint('画像存在確認失敗: $imagePath');
      return false;
    } catch (e) {
      debugPrint('画像存在確認エラー: $e');
      return false;
    }
  }
  
  /// Supabaseの手書きサンプルデータをCaptureDataに変換
  /// 
  /// [supabaseData] - Supabaseから取得したデータ
  /// 
  /// Returns: CaptureDataオブジェクト
  CaptureData convertToCaptureData(Map<String, dynamic> supabaseData) {
    // 文字ラベルを抽出（新しいスキーマ対応）
    final List<String> characterLabels = [];
    try {
      // 認識された文字を追加
      final geminiChar = supabaseData['gemini_recognized_char'] as String?;
      if (geminiChar != null && geminiChar.isNotEmpty) {
        characterLabels.add(geminiChar);
      }
      
      // charactersテーブルからの文字情報も追加
      final characterInfo = supabaseData['characters'] as Map<String, dynamic>?;
      if (characterInfo != null) {
        final character = characterInfo['character'] as String?;
        if (character != null && character.isNotEmpty && !characterLabels.contains(character)) {
          characterLabels.add(character);
        }
      }
    } catch (e) {
      debugPrint('文字ラベル抽出エラー: $e');
    }
    
    // writer_numberを正しく取得
    String writerNumber = '';
    try {
      final writerInfo = supabaseData['writers'] as Map<String, dynamic>?;
      if (writerInfo != null) {
        writerNumber = writerInfo['writer_number'] as String? ?? '';
      }
    } catch (e) {
      debugPrint('記入者番号抽出エラー: $e');
    }
    
    return CaptureData(
      imagePath: supabaseData['image_path'] as String?,
      writerNumber: writerNumber,
      captureTime: DateTime.tryParse(supabaseData['created_at'] as String? ?? '') ?? DateTime.now(),
      characterLabels: characterLabels,
      isProcessed: supabaseData['score_white'] != null, // 評価スコアがあれば処理済み
    );
  }
}