# 🚦 Coding Activity 5 – Street Light Detection and Recognition for ADAS 🚦

## 📝 Description

An upcoming trend in ADAS technologies focuses on the detection and recognition of street lights. This is incredibly useful as drivers may sometimes miss or misinterpret traffic lights for various reasons.

## 🎯 Problem Statement

- Develop the Street Light Detection system in Rust.
- There are two approaches for this problem:
  1. **YOLO Approach**: Use the object detection algorithm from the previous activity. Filter out all object classes to focus solely on street lights. Further, apply a colour detection algorithm to determine which colour the light is currently displaying (RED, AMBER, GREEN) or if it is DEACTIVE.
  2. **Direct Deep Learning Approach**: Use a dataset of various street light statuses (if available) to train a CNN that directly outputs one of the four classes.
- Test the developed system on random road images and/or video feed.

I implemented **approach 1**, and added a model-free fallback so the activity also runs without the detector.

## 🛠 Technologies

- **Programming Language**: Rust
- **Detection**: YOLOX filtered to COCO class 9 (`traffic light`), or a model-free brightness-blob detector
- **Colour Recognition**: HSV weighting in `adas-core/src/traffic_light.rs` → `RED` / `AMBER` / `GREEN` / `OFF`
- **Temporal logic**: `LightTracker` (several frames must agree before the state changes)

## 🏁 Getting Started

```bash
./scripts/fetch-data.sh                                            # gets the YOLOX model
cargo run --release -p street_light_detection                      # YOLOX if downloaded, else model-free
cargo run --release -p street_light_detection -- --preset nano     # pick a specific YOLOX
cargo run --release -p street_light_detection -- --classical       # force the model-free detector
cargo run --release -p street_light_detection -- --input lights/ --verbose
```

`--input` accepts an image, a folder, or a video. `--confirm N` sets how many frames must agree before the confirmed state changes. If no YOLOX model has been downloaded, the model-free detector runs automatically (and the summary says so).

### How the colour read works

A traffic-light crop rarely contains just the lamp: there is housing, a pole and the sky behind. The classifier converts each pixel to HSV, ignores anything too dark to be a lit lamp, weights the rest by `saturation² × value`, and sums the weight per hue band (red / amber / green). If the total weight per pixel is below `LIT_DENSITY` the crop is `OFF`; otherwise the band with the most weight wins, and its share of the total becomes the confidence.

### Why a tracker, like Activity 3

One frame is not a decision. A flickering reflection or a single misread crop should not flip the display, so `LightTracker` only changes the confirmed state after `--confirm` consecutive frames agree, and falls back to no signal once the light has been out of view for that many frames.

### Results

![A green left-turn signal in Kumasi, boxed by YOLOX and read as GREEN 100%](../docs/lights.webp)

With `--preset nano` on Wikimedia street photos, the green left-turn signal in a Kumasi intersection is read as **GREEN 100%**, amber lamps in Inuvik as **AMBER**. YOLOX-Nano occasionally boxes a bright billboard as a "traffic light"; the colour read of that box is still consistent (an orange advert reads amber), and `--preset tiny`/`s` reduce the false positives.

## 📚 References

This activity expects the implementation to be carried out without direct help or direct code. See the References in Activity 4 for the detector and Activity 3 for the tracking idea.
