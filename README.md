# ♻️ YOLOv8 기반 실시간 쓰레기 분류 어플리케이션

YOLOv8 모델을 TFLite로 변환하여 Flutter 앱에서 실시간으로 쓰레기를 인식하고 분류하는 프로젝트입니다.

## ✨ 주요 기능
* **실시간 객체 탐지**: 카메라 스트림을 통한 즉각적인 쓰레기 인식
* **종류별 색상 구분**: 인식된 쓰레기마다 다른 색상의 박스 표시
  * 빨강: Trash
  * 주황: Cardboard
  * 노랑: Glass
  * 초록: Metal
  * 파랑: Paper
  * 보라: Plastic

## 🛠 기술 스택
* **AI 모델**: YOLOv8 (Ultralytics)
* **프레임워크**: Flutter
* **엔진**: TensorFlow Lite (tflite_flutter)
