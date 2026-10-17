# 🚗 Coding Activity 3 – Development of Speed Limit Recognition System for ADAS 🛑

## 📝 Description

Speed Limit Recognition is a specialized subset of Traffic Sign Recognition systems. Understanding the speed limits of different road sections is essential for road safety and to avoid penalties. This system uses AI algorithms in tandem with the front camera of the vehicle to identify speed limit signs and display them on the instrument panel.

## 🎯 Problem Statement

- Develop the Speed Limit Recognition system in Rust.
- Use computer vision techniques, along with machine learning or deep learning algorithms.
- Obtain a publicly available dataset containing speed limit sign images.
- Train a deep learning CNN model using the dataset, reserving some data for validation and testing.
- Upon successful validation with high accuracy, test the model on a test dataset and save it for future use.
- Further validate the system with different speed limit images or video feeds.

## 🛠 Technologies

- **Programming Language**: Rust
- **Inference**: the Activity 2 classifier ([tract](https://github.com/sonos/tract))
- **Logic**: `SpeedLimitTracker` in `adas-core/src/speed.rs`
- **Data Source**: GTSRB, or the built-in `--demo` drive past rendered signs

## 📚 References

- [Automatic Recognition of Speed Limit Signs: Deep Learning with Keras and TensorFlow](https://hackernoon.com/automatic-recognition-of-speed-limit-signs-deep-learning-with-keras-and-tensorflow-310d90af9826)
- [MDPI: Speed Limit Sign Detection](https://www.mdpi.com/2313-433X/3/3/25/htm)

## 🏁 Getting Started

```bash
cargo run --release -p speed_limit_recognition -- --demo           # no data needed
cargo run --release -p speed_limit_recognition -- frames/          # your crops, in file-name order
```

### Why a tracker

A classifier answers *"what is in this image?"*. A car needs *"what limit is in force now?"*. The tracker:

- ignores predictions under `--min-confidence`;
- changes the limit only when one reading has `--votes` of the last `--window` frames **and** leads the runner-up by `--margin`. This model reports ~100% confidence even when wrong, so one frame proves nothing;
- understands **end-of-limit** signs (GTSRB classes 6 and 32). The first version treated them as "not a speed sign", so a limit could never be lifted;
- keeps the old limit when readings flip-flop: missing a sign is safer than switching to a wrong one (tested in `a_flip_flopping_reading_keeps_the_old_limit`).

The demo prints every frame, the moment each limit changes, and whether each sign ended up right.
