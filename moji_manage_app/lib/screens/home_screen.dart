import 'package:flutter/material.dart';
import 'image_capture_screen.dart';
import 'sample_list_screen.dart';
import '../models/capture_data.dart';
import '../services/api_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final List<CaptureData> _recentCaptures = [];
  final ApiService _apiService = ApiService();
  
  // APIから取得したデータ
  Map<String, dynamic>? _stats;
  List<Map<String, dynamic>> _recentActivities = [];
  bool _isLoading = true;

  void _navigateToImageCapture() async {
    final result = await Navigator.push<CaptureData>(
      context,
      MaterialPageRoute(builder: (context) => const ImageCaptureScreen()),
    );

    if (result != null) {
      setState(() {
        _recentCaptures.insert(0, result);
      });
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('画像を取り込みました: ${result.writerNumber}'),
            backgroundColor: Colors.green,
          ),
        );
      }
    }
  }

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  /// データ読み込み
  Future<void> _loadData() async {
    setState(() {
      _isLoading = true;
    });

    try {
      // 統計情報と最近の活動を並列で取得
      final results = await Future.wait([
        _apiService.getStats(),
        _apiService.getRecentActivity(limit: 10),
      ]);

      setState(() {
        _stats = results[0] as Map<String, dynamic>?;
        _recentActivities = (results[1] as List<Map<String, dynamic>>?) ?? [];
        _isLoading = false;
      });
    } catch (e) {
      debugPrint('データ読み込み失敗: $e');
      setState(() {
        _isLoading = false;
      });
    }
  }

  /// APIヘルスチェック実行
  Future<void> _checkApiHealth() async {
    try {
      final isHealthy = await _apiService.checkHealth();
      
      if (mounted) {
        if (isHealthy) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('✅ APIサーバー接続OK'),
              backgroundColor: Colors.green,
              duration: Duration(seconds: 3),
            ),
          );
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('❌ APIサーバー接続NG - サーバーが停止しているか確認してください'),
              backgroundColor: Colors.red,
              duration: Duration(seconds: 5),
            ),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('❌ APIヘルスチェック失敗: $e'),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 5),
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('📝 美文字データ管理'),
        backgroundColor: Colors.blue.shade50,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildStatisticsCard(),
            const SizedBox(height: 20),
            _buildMainFunctions(),
            const SizedBox(height: 20),
            _buildRecentActivity(),
          ],
        ),
      ),
    );
  }

  Widget _buildStatisticsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Text(
                  '📊 統計情報',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const Spacer(),
                if (_isLoading)
                  const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                else
                  IconButton(
                    icon: const Icon(Icons.refresh, size: 20),
                    onPressed: _loadData,
                    tooltip: '更新',
                  ),
              ],
            ),
            const SizedBox(height: 12),
            if (_isLoading)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(20.0),
                  child: Text('読み込み中...', style: TextStyle(color: Colors.grey)),
                ),
              )
            else if (_stats != null)
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _buildStatItem(
                    '総サンプル数',
                    '${_stats!['stats']?['samples_count'] ?? 0}件',
                    Colors.blue,
                  ),
                  _buildStatItem(
                    '評価済み',
                    '${_stats!['stats']?['approved_samples'] ?? 0}件',
                    Colors.green,
                  ),
                  _buildStatItem(
                    '記入者数',
                    '${_stats!['stats']?['writers_count'] ?? 0}人',
                    Colors.purple,
                  ),
                ],
              )
            else
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(20.0),
                  child: Text(
                    'データを取得できませんでした\nAPIサーバーが起動しているか確認してください',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatItem(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          value,
          style: TextStyle(
            fontSize: 20,
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

  Widget _buildMainFunctions() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '🔧 主要機能',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: _buildFunctionButton(
                    icon: Icons.camera_alt,
                    title: '📷 新規撮影',
                    subtitle: '開始',
                    onTap: _navigateToImageCapture,
                    color: Colors.blue,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildFunctionButton(
                    icon: Icons.list_alt,
                    title: '📋 一覧管理',
                    subtitle: '・評価',
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const SampleListScreen(),
                        ),
                      );
                    },
                    color: Colors.green,
                  ),
                ),
              ],
            ),
            /*const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: _buildFunctionButton(
                icon: Icons.health_and_safety,
                title: '🔧 APIヘルスチェック',
                subtitle: 'サーバー接続確認',
                onTap: _checkApiHealth,
                color: Colors.orange,
              ),
            ),*/
          ],
        ),
      ),
    );
  }

  Widget _buildFunctionButton({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
    required Color color,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          border: Border.all(color: color.withOpacity(0.3)),
          borderRadius: BorderRadius.circular(8),
          color: color.withOpacity(0.1),
        ),
        child: Column(
          children: [
            Icon(icon, size: 32, color: color),
            const SizedBox(height: 8),
            Text(
              title,
              style: TextStyle(fontWeight: FontWeight.bold, color: color),
              textAlign: TextAlign.center,
            ),
            Text(
              subtitle,
              style: TextStyle(fontSize: 12, color: color.withOpacity(0.8)),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRecentActivity() {
    return Expanded(
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Text(
                    '📈 最近の活動',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  const Spacer(),
                  if (!_isLoading)
                    IconButton(
                      icon: const Icon(Icons.refresh, size: 20),
                      onPressed: _loadData,
                      tooltip: '更新',
                    ),
                ],
              ),
              const SizedBox(height: 12),
              Expanded(
                child: _isLoading
                    ? const Center(
                        child: CircularProgressIndicator(),
                      )
                    : _recentActivities.isEmpty
                        ? const Center(
                            child: Text(
                              'まだデータがありません\n新規撮影から開始してください',
                              textAlign: TextAlign.center,
                              style: TextStyle(color: Colors.grey),
                            ),
                          )
                        : ListView.separated(
                            itemCount: _recentActivities.length,
                            separatorBuilder: (context, index) => const Divider(),
                            itemBuilder: (context, index) {
                              final activity = _recentActivities[index];
                              final hasScores = activity['has_scores'] == true;
                              final createdAt = DateTime.tryParse(activity['created_at'] ?? '');
                              final formattedDate = createdAt != null
                                  ? '${createdAt.month}/${createdAt.day} ${createdAt.hour}:${createdAt.minute.toString().padLeft(2, '0')}'
                                  : '不明';
                              
                              return ListTile(
                                leading: Icon(
                                  hasScores ? Icons.check_circle : Icons.pending,
                                  color: hasScores ? Colors.green : Colors.orange,
                                ),
                                title: Text('${activity['writer_number']} - ${activity['character']}'),
                                subtitle: Text(
                                  '${hasScores ? '✅ 評価済み' : '⏳ 未評価'} • 📅 $formattedDate',
                                  style: TextStyle(
                                    color: hasScores ? Colors.green.shade700 : Colors.orange.shade700,
                                  ),
                                ),
                                trailing: const Icon(Icons.arrow_forward_ios, size: 16),
                                onTap: () {
                                  Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                      builder: (context) => const SampleListScreen(),
                                    ),
                                  );
                                },
                              );
                            },
                          ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}