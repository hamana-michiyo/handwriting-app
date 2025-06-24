class CaptureData {
  final String? imagePath;
  final String writerNumber;
  final DateTime captureTime;
  final List<String> characterLabels;
  final bool isProcessed;

  CaptureData({
    this.imagePath,
    required this.writerNumber,
    required this.captureTime,
    this.characterLabels = const [],
    this.isProcessed = false,
  });

  CaptureData copyWith({
    String? imagePath,
    String? writerNumber,
    DateTime? captureTime,
    List<String>? characterLabels,
    bool? isProcessed,
  }) {
    return CaptureData(
      imagePath: imagePath ?? this.imagePath,
      writerNumber: writerNumber ?? this.writerNumber,
      captureTime: captureTime ?? this.captureTime,
      characterLabels: characterLabels ?? this.characterLabels,
      isProcessed: isProcessed ?? this.isProcessed,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'imagePath': imagePath,
      'writerNumber': writerNumber,
      'captureTime': captureTime.toIso8601String(),
      'characterLabels': characterLabels,
      'isProcessed': isProcessed,
    };
  }

  factory CaptureData.fromJson(Map<String, dynamic> json) {
    return CaptureData(
      imagePath: json['imagePath'],
      writerNumber: json['writerNumber'],
      captureTime: DateTime.parse(json['captureTime']),
      characterLabels: List<String>.from(json['characterLabels'] ?? []),
      isProcessed: json['isProcessed'] ?? false,
    );
  }
}