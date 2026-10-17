# 🧩 Coding Activity 7 – Let's Put All Together 🧩

## 📝 Description

This is the culmination of all the previous coding activities. By this point, we have individually developed:
- Lane Detection
- Traffic Sign Recognition
- Speed Limit Recognition
- Street Lamp Detection and Recognition
- Instrument Cluster GUI

Now, the objective is to integrate all these features into one cohesive system.

## 🎯 Problem Statement

- Use Rust for development. Consider using Object-Oriented Programming for better integration.
- Take various video feeds as input to your system.
- Adapt all the previously implemented algorithms to work in conjunction. 
- The aim is to detect lanes, traffic signs, speed limits, and street lamps, and display them on the GUI in real-time as they are detected in the video feed.
- Test the complete system with multiple video feeds.

## 🛠 Technologies

- **Programming Language**: Rust
- **Libraries**: everything in `adas-core`: lanes, the YOLOX object detector and the ONNX sign classifier (both on tract), the speed-limit and traffic-light trackers, the cluster renderer, ffmpeg video I/O

![Lanes, YOLOX object detection, speed-limit and traffic-light recognition, and the instrument cluster on a dashcam clip](../docs/drive.webp)
  
## 🏁 Getting Started

```bash
./scripts/fetch-data.sh
cargo run --release -p adas_drive                                       # → output/drive.mp4
cargo run --release -p adas_drive -- --signs "2:30,9:80,16:70" --lights "5:red,12:green" --output output/drive.webp
```

### What happens each frame

1. **Lane detection** and **YOLOX object detection** run on the clean camera image (Activities 1 and 4). The detection is real: the boxes land on the cars and trucks that are already in the clip.
2. A **roadside speed sign** and a **traffic light** are painted into the scene, growing as the car approaches. The sample clips are from a US highway with no German signs and no signals, so both are simulated, and the video says so.
3. **The real classifiers read them from the frame's own pixels** (Activities 2 and 5). They see exactly what is on screen, with no hint of which sign or colour it is.
4. The **speed-limit and traffic-light trackers** confirm them over several frames (Activity 3).
5. The **instrument cluster** is drawn below the video (Activity 6). It shows a simulated driver who reacts to each new limit about a second and a half late, so the over-speed warning gets a moment to shine.
6. The camera view carries the detection boxes, the lane shading, the recognised sign and the confirmed signal.

`--signs` takes `seconds:km/h` pairs (`0` = an "end of 80" sign). `--lights` takes `seconds:colour` pairs (`red`, `amber` or `green`). Signs the model reads reliably when rendered: 20, 30, 70, 80 and mostly 120 (see the root README).

