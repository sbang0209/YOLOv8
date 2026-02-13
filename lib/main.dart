import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image/image.dart' as img; // 이미지 처리를 위해 필수
import 'dart:typed_data';

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Permission.camera.request();
  _cameras = await availableCameras();
  runApp(const MyApp());
}

class Recognition {
  final Rect location;
  final String label;
  final double score;
  final int classId;

  Recognition(this.location, this.label, this.score, this.classId);
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '쓰레기 분류 AI',
      theme: ThemeData.dark(),
      home: const TrashDetectionScreen(),
    );
  }
}

class TrashDetectionScreen extends StatefulWidget {
  const TrashDetectionScreen({super.key});
  @override
  State<TrashDetectionScreen> createState() => _TrashDetectionScreenState();
}

class _TrashDetectionScreenState extends State<TrashDetectionScreen> {
  CameraController? _controller;
  Interpreter? _interpreter;
  List<Recognition> _results = [];
  bool _isDetecting = false;
  bool _isModelLoaded = false;

  final List<String> _labels = ['Trash', 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic'];

  @override
  void initState() {
    super.initState();
    _initApp();
  }

  Future<void> _initApp() async {
    await _loadModel();
    await _initCamera();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/best_float32.tflite');
      setState(() { _isModelLoaded = true; });
      debugPrint("✅ 모델 로드 성공");
    } catch (e) {
      debugPrint("❌ 모델 로드 실패: $e");
    }
  }

  Future<void> _initCamera() async {
    if (_cameras.isEmpty) return;
    _controller = CameraController(_cameras[0], ResolutionPreset.medium, enableAudio: false);
    await _controller!.initialize();

    _controller!.startImageStream((CameraImage image) {
      if (_isModelLoaded && !_isDetecting) {
        _isDetecting = true;
        _runInference(image);
      }
    });
    if (mounted) setState(() {});
  }

  Color getBoxColor(int classId) {
    switch (classId) {
      case 0: return Colors.red;
      case 1: return Colors.orange;
      case 2: return Colors.yellow;
      case 3: return Colors.green;
      case 4: return Colors.blue;
      case 5: return Colors.purple;
      default: return Colors.white;
    }
  }

  // [추가된 생략 코드] 이미지 전처리 및 추론
  void _runInference(CameraImage cameraImage) async {
    if (_interpreter == null) return;

    // 1. 카메라 이미지(YUV)를 RGB로 변환하고 640x640으로 리사이즈
    var image = _convertCameraImage(cameraImage);
    var resizedImage = img.copyResize(image, width: 640, height: 640);

    // 2. 모델 입력용 4차원 리스트 생성 [1, 640, 640, 3]
    var input = List.generate(1, (i) =>
        List.generate(640, (y) =>
            List.generate(640, (x) =>
                List.generate(3, (c) {
                  // 정규화: 0~255 값을 0.0~1.0 사이로 변환
                  var pixel = resizedImage.getPixel(x, y);
                  if (c == 0) return pixel.r / 255.0;
                  if (c == 1) return pixel.g / 255.0;
                  return pixel.b / 255.0;
                })
            )
        )
    );

    var output = List.filled(1 * 10 * 8400, 0.0).reshape([1, 10, 8400]);

    // 3. 모델 실행
    _interpreter!.run(input, output);

    // 4. 후처리
    List<Recognition> detections = _postProcess(output[0]);

    if (mounted) {
      setState(() {
        _results = detections;
        _isDetecting = false;
      });
    }
  }

  // [추가된 생략 코드] 카메라 YUV 포맷을 RGB 이미지 객체로 변환
  img.Image _convertCameraImage(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final img.Image res = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex = (y / 2).floor() * (width / 2).floor() + (x / 2).floor();
        final int index = y * width + x;

        final int yp = image.planes[0].bytes[index];
        final int up = image.planes[1].bytes[uvIndex];
        final int vp = image.planes[2].bytes[uvIndex];

        // YUV to RGB 변환 공식
        int r = (yp + 1.402 * (vp - 128)).round().clamp(0, 255);
        int g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128)).round().clamp(0, 255);
        int b = (yp + 1.772 * (up - 128)).round().clamp(0, 255);

        res.setPixelRgb(x, y, r, g, b);
      }
    }
    return res;
  }

  List<Recognition> _postProcess(List<List<double>> output) {
    List<Recognition> candidates = [];
    for (int i = 0; i < 8400; i++) {
      double maxScore = 0;
      int classId = -1;
      for (int c = 0; c < 6; c++) {
        if (output[c + 4][i] > maxScore) {
          maxScore = output[c + 4][i];
          classId = c;
        }
      }

      if (maxScore > 0.45) {
        double x = output[0][i];
        double y = output[1][i];
        double w = output[2][i];
        double h = output[3][i];

        candidates.add(Recognition(
          Rect.fromCenter(center: Offset(x, y), width: w, height: h),
          _labels[classId],
          maxScore,
          classId,
        ));
      }
    }

    List<Recognition> finalResults = [];
    candidates.sort((a, b) => b.score.compareTo(a.score));
    while (candidates.isNotEmpty) {
      var best = candidates.removeAt(0);
      finalResults.add(best);
      candidates.removeWhere((item) => _calculateIoU(best.location, item.location) > 0.5);
    }
    return finalResults;
  }

  double _calculateIoU(Rect a, Rect b) {
    var intersection = a.intersect(b);
    if (intersection.width <= 0 || intersection.height <= 0) return 0.0;
    double intersectionArea = intersection.width * intersection.height;
    double unionArea = (a.width * a.height) + (b.width * b.height) - intersectionArea;
    return intersectionArea / unionArea;
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final size = MediaQuery.of(context).size;

    return Scaffold(
      appBar: AppBar(title: const Text("쓰레기 분류 AI")),
      body: Stack(
        children: [
          SizedBox(
            width: size.width,
            height: size.height,
            child: CameraPreview(_controller!),
          ),
          ..._results.map((reco) {
            // 좌표 스케일링 (중요: YOLO 결과값은 0~640 사이의 값임)
            double left = reco.location.left * size.width / 640;
            double top = reco.location.top * size.height / 640;
            double width = reco.location.width * size.width / 640;
            double height = reco.location.height * size.height / 640;

            return Positioned(
              left: left,
              top: top,
              child: Container(
                width: width,
                height: height,
                decoration: BoxDecoration(
                  border: Border.all(color: getBoxColor(reco.classId), width: 3),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Container(
                  color: getBoxColor(reco.classId).withValues(alpha: 0.5),
                  padding: const EdgeInsets.symmetric(horizontal: 4),
                  child: Text(
                    "${reco.label} ${(reco.score * 100).toStringAsFixed(0)}%",
                    style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 12),
                  ),
                ),
              ),
            );
          }),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    _interpreter?.close();
    super.dispose();
  }
}