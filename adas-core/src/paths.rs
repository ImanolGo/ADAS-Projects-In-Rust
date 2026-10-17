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
