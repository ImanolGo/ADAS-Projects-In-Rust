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
- **Libraries**: everything in `adas-core`: lanes, the ONNX sign classifier (tract), the speed-limit tracker, the cluster renderer, ffmpeg video I/O

![Lanes, speed-limit recognition and the cluster on a dashcam clip](../docs/drive.webp)
  
## 🏁 Getting Started

```bash
./scripts/fetch-data.sh
cargo run --release -p adas_drive                                       # → output/drive.mp4
cargo run --release -p adas_drive -- --signs "2:30,9:80,16:70" --output output/drive.webp
```

### What happens each frame

1. **Lane detection** on the camera image (Activity 1).
2. A **roadside speed sign** is painted into the scene, growing as the car approaches. The sample clips are from a US highway with no German signs, so the signs are simulated, and the video says so.
3. **The real classifier reads the sign from the frame's pixels** (Activity 2). It sees exactly what is on screen, and gets no hint of which sign it is.
4. The **speed-limit tracker** confirms it over several frames (Activity 3).
5. The **instrument cluster** is drawn below the video (Activity 6). It shows a simulated driver who reacts to each new limit about a second and a half late, so the over-speed warning gets a moment to shine.

`--signs` takes `seconds:km/h` pairs (`0` = an "end of 80" sign). Signs the model reads reliably when rendered: 20, 30, 70, 80 and mostly 120 (see the root README).

