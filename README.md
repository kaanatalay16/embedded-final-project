# EE 4065 - Embedded Digital Image Processing Final Project

**Student**: KAAN ATALAY  
**ID**: 150720057  
**Date**: January 2026

## 📋 Project Overview

This project implements various embedded digital image processing algorithms on the ESP32-CAM module, including:

1. **Size-Based Thresholding** - Extract objects based on pixel count
2. **YOLO Digit Detection** - Real-time handwritten digit detection
3. **Image Scaling** - Upsampling/downsampling with non-integer factors
4. **Multi-Model Recognition** - SqueezeNet, MobileNet, EfficientNet, ResNet
5. **BONUS: FOMO & SSD** - Lightweight object detection
6. **BONUS: MobileViT** - Vision Transformer for embedded systems

## 🗂️ Project Structure

```
embedded-project/
│
├── q1_thresholding/          # Question 1: Size-based thresholding
│   ├── thresholding_pc.py    # Python implementation (PC)
│   └── esp32_thresholding.ino # ESP32-CAM implementation
│
├── q2_yolo_digit/            # Question 2: YOLO digit detection
│   ├── train_yolo_digit.py   # Training script
│   ├── esp32_yolo_inference.ino
│   └── digit_model.h         # Model placeholder
│
├── q3_sampling/              # Question 3: Up/Downsampling
│   └── esp32_sampling.ino
│
├── q4_multi_model/           # Question 4: Multi-model recognition
│   ├── train_multi_models.py
│   ├── esp32_multi_model.ino
│   └── *_model.h             # Model headers
│
├── q5_bonus_fomo_ssd/        # Question 5 BONUS: FOMO & SSD
│   ├── train_fomo.py
│   ├── train_ssd_mobilenet.py
│   └── esp32_fomo_ssd.ino
│
├── q6_bonus_mobilevit/       # Question 6 BONUS: MobileViT
│   ├── train_mobilevit.py
│   └── esp32_mobilevit.ino
│
├── report/                   # LaTeX report
│   └── main.tex
│
├── cheat_sheet/              # Presentation preparation
│   ├── CHEAT_SHEET_TR.md
│   └── CHEAT_SHEET_EN.md
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites

- **Hardware**: ESP32-CAM (AI Thinker)
- **Software**: 
  - Arduino IDE with ESP32 board support
  - Python 3.8+
  - TensorFlow 2.10+

### Python Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Arduino IDE Setup

1. Install ESP32 board support
2. Select board: **AI Thinker ESP32-CAM**
3. Install required libraries:
   - esp32-camera
   - TensorFlow Lite Micro (tflite-micro)

### Training Models

```bash
# Question 2: YOLO
cd q2_yolo_digit
python train_yolo_digit.py

# Question 4: Multi-model
cd q4_multi_model
python train_multi_models.py

# Question 5: FOMO & SSD
cd q5_bonus_fomo_ssd
python train_fomo.py
python train_ssd_mobilenet.py

# Question 6: MobileViT
cd q6_bonus_mobilevit
python train_mobilevit.py
```

### Converting Models to C Headers

```bash
# After training, convert TFLite to C header
xxd -i model.tflite > model.h
```

### Uploading to ESP32-CAM

1. Open `.ino` file in Arduino IDE
2. Connect ESP32-CAM via USB-TTL converter
3. Put in download mode (GPIO0 → GND)
4. Upload sketch
5. Remove GPIO0 connection and reset

## 📊 Performance Summary

| Question | Algorithm | Inference Time | Memory Usage |
|----------|-----------|----------------|--------------|
| Q1 | Binary Search Threshold | 15-20ms | 76KB |
| Q2 | YOLOv8-nano | ~100ms | 300KB |
| Q3 | Bilinear 2x | 85ms | 614KB |
| Q4 | Multi-model Ensemble | 200ms | 200KB/model |
| Q5 | FOMO | 50ms | 100KB |
| Q5 | SSD+MobileNet | 120ms | 250KB |
| Q6 | MobileViT | 150ms | 350KB |

## 📝 Report Compilation

```bash
cd report
/Library/TeX/texbin/pdflatex main.tex
```

## 🔗 References

- [STMicroelectronics AI Model Zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo)
- [Edge Impulse FOMO](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/object-detection/fomo-object-detection-for-constrained-devices)
- [MobileViT Keras Example](https://keras.io/examples/vision/mobilevit/)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)

## 📄 License

This project is submitted as part of EE 4065 course requirements.

---

**Good luck with the presentation! 🍀**
