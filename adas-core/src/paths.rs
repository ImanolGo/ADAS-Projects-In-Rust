//! Where data lives, independent of the directory you run from.
//!
//! The original binaries opened `../data/...` relative to the *current*
//! directory, so they only worked when started from inside an activity folder.

use std::path::PathBuf;

/// The repository's `data/` folder, or `$ADAS_DATA_DIR` if set.
pub fn data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("ADAS_DATA_DIR") {
        return PathBuf::from(dir);
    }
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("data");
    dir.canonicalize().unwrap_or(dir)
}

/// The bundled GTSRB classifier (Keras CNN exported to ONNX).
pub fn default_model() -> PathBuf {
    data_dir().join("Models").join("traffic-sign-recognition-model.onnx")
}

/// Directory holding the YOLOX ONNX models (`scripts/fetch-data.sh`).
pub fn detector_dir() -> PathBuf {
    data_dir().join("Models").join("yolo")
}

/// The YOLOX-Nano detector: ~4 MB, fast enough for video on a CPU.
pub fn default_detector() -> PathBuf {
    detector_dir().join("yolox_nano.onnx")
}

/// A named YOLOX preset: `nano` (416), `tiny` (416) or `s` (640).
pub fn detector_preset(name: &str) -> Option<PathBuf> {
    let file = match name {
        "nano" => "yolox_nano.onnx",
        "tiny" => "yolox_tiny.onnx",
        "s" | "small" => "yolox_s.onnx",
        _ => return None,
    };
    Some(detector_dir().join(file))
}

/// The dashcam clip used by the lane and integration demos (see `scripts/fetch-data.sh`).
pub fn default_video() -> PathBuf {
    data_dir().join("video.mp4")
}

/// GTSRB test images (Kaggle layout: `Test/*.png` plus `Test.csv`).
pub fn default_test_dir() -> PathBuf {
    data_dir().join("Test")
}

/// Fonts are compiled in, so rendering never depends on system fonts.
pub static FONT_SEMIBOLD: &[u8] = include_bytes!("../../assets/fonts/BarlowSemiCondensed-SemiBold.ttf");
pub static FONT_BOLD: &[u8] = include_bytes!("../../assets/fonts/BarlowSemiCondensed-Bold.ttf");
