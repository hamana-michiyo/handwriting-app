import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';

/// 手書き文字評価APIとの通信を担当するサービスクラス
class ApiService {
  // API基本設定 - 環境別URL
  static const String _developmentUrl = 'http://192.168.1.68:8001';
  static const String _productionUrl = 'https://handwriting-app-qqp3.onrender.com';
  
  // 現在の環境に応じたベースURL
  static String get _baseUrl {
    if (kDebugMode) {
      // デバッグモード（開発環境）
      return _developmentUrl;
      // return _productionUrl; // 本番環境に切り替え
    } else {
      // リリースモード（本番環境）
      return _productionUrl;
    }
  }
  
  static const Duration _timeout = Duration(seconds: 120);
  
  // エンドポイント
  static const String _processFormEndpoint = '/process-form';
  static const String _processCroppedFormEndpoint = '/process-cropped-form';
  static const String _healthEndpoint = '/health';
  static const String _statsEndpoint = '/stats';
  static const String _recentActivityEndpoint = '/recent-activity';
  
  /// APIサーバーのヘルスチェック
  Future<bool> checkHealth() async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl$_healthEndpoint'),
        headers: {'Content-Type': 'application/json'},
      ).timeout(_timeout);
      
      return response.statusCode == 200;
    } catch (e) {
      debugPrint('ヘルスチェック失敗: $e');
      return false;
    }
  }
  
  /// 記入用紙画像を処理してサーバーに送信
  /// 
  /// [imagePath] - 処理する画像ファイルのパス
  /// [writerNumber] - 記入者番号
  /// [writerAge] - 記入者年齢（オプション）
  /// [writerGrade] - 記入者学年（オプション）
  /// [autoSave] - 自動保存フラグ
  /// 
  /// Returns: 処理結果のMap、エラー時はnull
  Future<Map<String, dynamic>?> processFormImage({
    required String imagePath,
    required String writerNumber,
    int? writerAge,
    String? writerGrade,
    bool autoSave = true,
  }) async {
    try {
      // 画像ファイルを読み込み
      final imageFile = File(imagePath);
      if (!await imageFile.exists()) {
        debugPrint('画像ファイルが存在しません: $imagePath');
        return null;
      }
      
      // 画像をBase64エンコード
      final imageBytes = await imageFile.readAsBytes();
      final base64Image = 'data:image/jpeg;base64,${base64Encode(imageBytes)}';
      
      // リクエストボディ作成
      final requestBody = {
        'image_base64': base64Image,
        'writer_number': writerNumber,
        'auto_save': autoSave,
      };
      
      // オプション項目を追加
      if (writerAge != null) {
        requestBody['writer_age'] = writerAge;
      }
      if (writerGrade != null) {
        requestBody['writer_grade'] = writerGrade;
      }
      
      debugPrint('API送信開始: $_baseUrl$_processFormEndpoint');
      debugPrint('記入者: $writerNumber, 年齢: $writerAge, 学年: $writerGrade');
      
      // API呼び出し
      final response = await http.post(
        Uri.parse('$_baseUrl$_processFormEndpoint'),
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: jsonEncode(requestBody),
      ).timeout(_timeout);
      
      debugPrint('APIレスポンス: ${response.statusCode}');
      
      // レスポンス処理
      if (response.statusCode == 200) {
        final responseData = jsonDecode(response.body) as Map<String, dynamic>;
        debugPrint('API処理成功: ${responseData['message']}');
        return responseData;
      } else {
        debugPrint('APIエラー: ${response.statusCode}');
        debugPrint('エラー内容: ${response.body}');
        return null;
      }
      
    } catch (e) {
      debugPrint('API通信エラー: $e');
      return null;
    }
  }
  
  /// 切り取り済み記入用紙画像を処理してサーバーに送信
  /// 
  /// [imagePath] - 切り取り済み画像ファイルのパス
  /// [writerNumber] - 記入者番号
  /// [writerAge] - 記入者年齢（オプション）
  /// [writerGrade] - 記入者学年（オプション）
  /// [autoSave] - 自動保存フラグ
  /// 
  /// Returns: 処理結果のMap（四隅検出、透視変換、認識結果含む）、エラー時はnull
  Future<Map<String, dynamic>?> processCroppedFormImage({
    required String imagePath,
    required String writerNumber,
    int? writerAge,
    String? writerGrade,
    bool autoSave = true,
  }) async {
    try {
      // 画像ファイルを読み込み
      final imageFile = File(imagePath);
      if (!await imageFile.exists()) {
        debugPrint('画像ファイルが存在しません: $imagePath');
        return null;
      }
      
      // 画像をBase64エンコード
      final imageBytes = await imageFile.readAsBytes();
      final base64Image = 'data:image/jpeg;base64,${base64Encode(imageBytes)}';
      
      // リクエストボディ作成
      final requestBody = {
        'image_base64': base64Image,
        'writer_number': writerNumber,
        'auto_save': autoSave,
      };
      
      // オプション項目を追加
      if (writerAge != null) {
        requestBody['writer_age'] = writerAge;
      }
      if (writerGrade != null) {
        requestBody['writer_grade'] = writerGrade;
      }
      
      debugPrint('✂️ 切り取り画像API送信開始: $_baseUrl$_processCroppedFormEndpoint');
      debugPrint('📐 処理内容: 四隅検出→透視変換→AI認識');
      debugPrint('記入者: $writerNumber, 年齢: $writerAge, 学年: $writerGrade');
      
      // API呼び出し
      final response = await http.post(
        Uri.parse('$_baseUrl$_processCroppedFormEndpoint'),
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: jsonEncode(requestBody),
      ).timeout(_timeout);
      
      debugPrint('APIレスポンス: ${response.statusCode}');
      
      // レスポンス処理
      if (response.statusCode == 200) {
        final responseData = jsonDecode(response.body) as Map<String, dynamic>;
        debugPrint('🎯 画像処理成功: ${responseData['message']}');
        if (responseData['perspective_corrected'] == true) {
          debugPrint('📐 透視変換完了: 完璧A4スキャン化');
        }
        return responseData;
      } else {
        debugPrint('APIエラー: ${response.statusCode}');
        debugPrint('エラー内容: ${response.body}');
        return null;
      }
      
    } catch (e) {
      debugPrint('切り取り画像API通信エラー: $e');
      return null;
    }
  }
  
  /// 統計情報を取得
  Future<Map<String, dynamic>?> getStats() async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl$_statsEndpoint'),
        headers: {'Content-Type': 'application/json'},
      ).timeout(_timeout);
      
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        debugPrint('統計情報取得エラー: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      debugPrint('統計情報取得失敗: $e');
      return null;
    }
  }
  
  /// 最近の活動を取得
  Future<List<Map<String, dynamic>>?> getRecentActivity({int limit = 10}) async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl$_recentActivityEndpoint?limit=$limit'),
        headers: {'Content-Type': 'application/json'},
      ).timeout(_timeout);
      
      if (response.statusCode == 200) {
        final responseData = jsonDecode(response.body) as Map<String, dynamic>;
        if (responseData['success'] == true) {
          return List<Map<String, dynamic>>.from(responseData['activities'] ?? []);
        }
        return null;
      } else {
        debugPrint('最近の活動取得エラー: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      debugPrint('最近の活動取得失敗: $e');
      return null;
    }
  }
  
  /// API結果からの認識文字を抽出
  List<String> extractRecognizedCharacters(Map<String, dynamic> apiResult) {
    final characters = <String>[];
    
    try {
      final characterResults = apiResult['character_results'] as List<dynamic>?;
      if (characterResults != null) {
        for (final result in characterResults) {
          final resultMap = result as Map<String, dynamic>;
          final geminiResult = resultMap['gemini_result'] as Map<String, dynamic>?;
          if (geminiResult != null) {
            final character = geminiResult['character'] as String?;
            if (character != null && character.isNotEmpty) {
              characters.add(character);
            }
          }
        }
      }
    } catch (e) {
      debugPrint('文字抽出エラー: $e');
    }
    
    return characters;
  }
  
  /// API結果からの数字認識結果を抽出
  Map<String, String> extractRecognizedNumbers(Map<String, dynamic> apiResult) {
    final numbers = <String, String>{};
    
    try {
      final numberResults = apiResult['number_results'] as List<dynamic>?;
      if (numberResults != null) {
        for (final result in numberResults) {
          final resultMap = result as Map<String, dynamic>;
          final name = resultMap['name'] as String?;
          final text = resultMap['text'] as String?;
          if (name != null && text != null && text.isNotEmpty) {
            numbers[name] = text;
          }
        }
      }
    } catch (e) {
      debugPrint('数字抽出エラー: $e');
    }
    
    return numbers;
  }
  
  /// API処理成功判定
  bool isSuccess(Map<String, dynamic>? apiResult) {
    if (apiResult == null) return false;
    return apiResult['success'] == true;
  }
  
  /// API結果からメッセージを取得
  String getMessage(Map<String, dynamic>? apiResult) {
    if (apiResult == null) return 'API通信に失敗しました';
    return apiResult['message'] as String? ?? '不明なエラーが発生しました';
  }
}