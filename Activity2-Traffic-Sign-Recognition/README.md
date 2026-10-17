# 🚦 Coding Activity 2 – Development of Traffic Sign Recognition System for ADAS 🚸

## 📝 Description

This coding activity focuses on the development of a Traffic Sign Recognition system, an important ADAS functionality. It aims to mitigate the risk of drivers missing crucial traffic signs during their journey. The system uses AI algorithms in conjunction with the front camera of the vehicle to detect and display traffic signs on the instrument cluster.

## 🎯 Problem Statement

- Develop the Traffic Sign Recognition system in Rust.
- Use computer vision and appropriate machine learning or deep learning algorithms.
- Download a publicly available dataset containing traffic sign images.
- Develop and train a deep learning CNN model, setting aside some data for validation and testing.
- After achieving high accuracy through training and validation, test the model on the test dataset and save it for later use.

## 🛠 Technologies

- **Programming Language**: Rust
- **Computer Vision**: [image](https://crates.io/crates/image) (crop, bilinear resize)
- **Inference**: [tract](https://github.com/sonos/tract), pure Rust ONNX runtime
- **Data Source**: [GTSRB](https://benchmark.ini.rub.de/)
- **Model**: pretrained Keras CNN (from the now-removed `sonu275981/GTSRB---German-Traffic-Sign-Recognition` repository), exported to ONNX: `data/Models/traffic-sign-recognition-model.onnx`

## 📚 References

- [Python Project: Traffic Signs Recognition](https://data-flair.training/blogs/python-project-traffic-signs-recognition/)
- [Class Data Science Project: Traffic Signs Recognition](https://medium.com/dataflair/class-data-science-project-for-2020-traffic-signs-recognition-12b09c131742)
- [Recognizing Traffic Signs with Over 98% Accuracy](https://towardsdatascience.com/recognizing-traffic-signs-with-over-98-accuracy-using-deep-learning86737aedc2ab)
- [Traffic Sign Classification with Keras and Deep Learning](https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/)

## 🏁 Getting Started

```bash
cargo run --release -p traffic_sign_recognition                         # data/Test + data/Test.csv
cargo run --release -p traffic_sign_recognition -- path/to/Test --gallery output/gallery.png
cargo run --release -p traffic_sign_recognition -- my_sign.jpg
```

With GTSRB's Kaggle layout (`Test/` next to `Test.csv`), each image is cropped to its annotated sign and scored against the true class: you get accuracy, the hardest classes and the most common confusions. `--gallery` saves a contact sheet with each answer framed green or red.

### Results

On the 51 GTSRB test images in this repository: **50/51 (98%)**, about 1 ms per sign. The original code got 48/51 on the same images (whole image, nearest-neighbour resize). On 21 photographed German signs: 21/21.

### What changed from the first version

- It did not compile on current Rust: `tract-onnx 0.20` pulls in `time 0.3.23`, which newer compilers reject. Now on tract 0.23, with `Cargo.lock` committed.
- `Test.csv` was never committed (`*.csv` was gitignored), so there was no ground truth and no accuracy. The labels for the bundled images are now in `data/Test.csv`.
- Paths were relative to the current directory, `argmax` panicked on NaN, and the label table was rebuilt for every image.
