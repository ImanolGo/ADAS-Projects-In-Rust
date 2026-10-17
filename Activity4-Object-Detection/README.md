# 📸 Coding Activity 4 – Object Detection 🚗🚶‍♂️

## 📝 Description

Object detection is crucial in ADAS systems for identifying objects that could lead to hazardous situations. Cameras are especially proficient at this task and are often used in conjunction with other sensors like Radar for data fusion. This activity aims to implement one of the state-of-the-art object detection algorithms, YOLO (You Only Look Once), known for its efficiency.

## 🎯 Problem Statement

- Develop the Object Detection system in Rust.
- Use a pretrained YOLO model and implement the algorithm to detect objects in road images.
- Test the algorithm on a variety of road images.
- Optionally, filter out unimportant object classes to focus on crucial road elements like cars, buses, pedestrians, bicycles, street lights, and traffic signs.
- Perform additional tests with these filters.

## 🔀 Why YOLOX instead of YOLOv3

The brief names YOLOv3. I used [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) instead, because it is a better fit for a pure-Rust, CPU pipeline while still being a single-shot YOLO-family detector:

| Model | Size | Compute | COCO mAP | This machine | Licence |
|---|---|---|---|---|---|
| YOLOv3 (Darknet) | 236 MB | 65.9 GFLOPs | 55.3 | needs a hand-written Darknet engine | public domain |
| YOLOv3-tiny | 34 MB | 5.6 GFLOPs | ~16.6 | same | public domain |
| **YOLOX-Nano** | **3.7 MB** | **1.08 GFLOPs** | 25.8 | **~40 ms/frame** | Apache-2.0 |
| YOLOX-Tiny | 20 MB | 6.45 GFLOPs | 32.8 | ~120 ms/frame | Apache-2.0 |
| YOLOX-S | 35 MB | 26.8 GFLOPs | 40.5 | slower | Apache-2.0 |

The official ONNX weights run directly under [tract](https://github.com/sonos/tract), the same pure-Rust runtime as the Activity 2 classifier, so there is **no Darknet engine to write and no Python to run**. YOLOX-Nano is roughly 2.7× faster than yolov3-tiny and ~9 mAP more accurate, in a ninth of the size. (YOLOv8/YOLO11 were rejected: their weights are AGPL-3.0, incompatible with this repo's MIT licence.)

## 🛠 Technologies

- **Programming Language**: Rust
- **Object Detection**: YOLOX (Nano / Tiny / S), ONNX run by tract
- **Labels**: COCO's 80 classes (`adas-core/src/coco.rs`), with an ADAS filter
- **Model Source**: [YOLOX releases](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0), downloaded by `scripts/fetch-data.sh`

## 📚 References

- [YOLOX: Exceeding YOLO Series in 2021 (arXiv)](https://arxiv.org/abs/2107.08430)
- [YOLOX GitHub](https://github.com/Megvii-BaseDetection/YOLOX)
- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [You Only Look Once: Unified, Real-Time Object Detection (arXiv)](https://arxiv.org/abs/1506.02640)

## 🏁 Getting Started

```bash
./scripts/fetch-data.sh                                   # downloads the YOLOX models (~60 MB)
cargo run --release -p object_detection                   # road clip → output/detect.mp4
cargo run --release -p object_detection -- --preset tiny  # more accurate, slower
cargo run --release -p object_detection -- --classes all --verbose
cargo run --release -p object_detection -- --input street.jpg --gallery output/detect.png
```

`--classes` takes `adas` (the default: vehicles, people, traffic lights and stop signs), `all`, or a comma-separated list such as `car,person,bicycle`. `--gallery` writes a contact sheet; `--input` accepts a single image, a folder, or a video.

### Results

![YOLOX-Nano on a dashcam clip: a car boxed at 47%](../docs/detect.webp)

On frames of the bundled dashcam clips YOLOX-Nano finds the cars and trucks in ~40 ms each on a CPU. Because it is the smallest model, it makes the occasional mistake (a bright billboard mistaken for a traffic light, for instance) — exactly the trade-off `--preset tiny` or `--preset s` is there to fix.
