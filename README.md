# Hardware-Efficient Hand Gesture Recognition on FPGA (HGR-Lite)

## Overview

This project implements a real-time hand gesture recognition system on FPGA using a custom lightweight CNN.

It covers the full pipeline:

* Model training (TensorFlow)
* Quantization (INT8)
* FPGA acceleration (Vitis HLS)
* Real-time inference (PYNQ + DMA)

---

## Key Features

* Depthwise CNN (MobileNet-style)
* INT8 quantized inference
* Custom FPGA accelerator (no prebuilt IP)
* AXI Stream + DMA pipeline
* Real-time webcam input
* Debug dashboard (Flask)

---

## System Pipeline

Camera → Preprocessing → DMA → FPGA CNN → DMA → CPU → Output

---

## Model Details

* Input: 64×64 RGB
* 4 Depthwise Conv Blocks
* BatchNorm + ReLU
* Fully Connected Layers
* Output: 6 gesture classes

---

## Hardware Implementation

* Vitis HLS (C++)
* BRAM for feature maps
* ROM for weights
* INT8 fixed-point arithmetic

---

## Deployment Flow

1. Train model in TensorFlow
2. Quantize weights to INT8
3. Export weights to C headers
4. Load into FPGA using HLS
5. Run inference using DMA on PYNQ

---

## Features

* Live gesture detection
* ROI selection (center / skin / edges)
* Confidence filtering
* Debug visualization dashboard

---

## Tech Stack

* Python, TensorFlow, OpenCV
* Vitis HLS
* PYNQ (Zynq FPGA)
* Flask

---
## Demo
![Pynq_Output](assets/PYNQZ2_Output.mp4)

## Future Improvements

* Better accuracy
* Faster inference (pipelining)
* Full hardware softmax
* IoT integration
