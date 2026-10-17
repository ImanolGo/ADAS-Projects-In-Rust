//! Integration tests for the YOLOX detector.
//!
//! The ONNX weights are downloaded by `scripts/fetch-data.sh` and are not
//! committed, so the model-dependent test skips when they are absent (a fresh
//! clone without the models should still `cargo test` green).

use adas_core::detector::ObjectDetector;
use adas_core::paths;
use image::{Rgb, RgbImage};

#[test]
fn missing_model_is_a_clear_error() {
    let err = match ObjectDetector::load("/no/such/model.onnx") {
        Ok(_) => panic!("loading a missing model should fail"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("fetch-data"), "unhelpful error: {err}");
}

#[test]
fn the_model_reads_its_square_input_size_and_stays_in_bounds() {
    let model = paths::default_detector();
    if !model.exists() {
        eprintln!("skipping: {} not downloaded (run scripts/fetch-data.sh)", model.display());
        return;
    }
    let detector = ObjectDetector::load(&model).unwrap();
    assert_eq!(detector.input_size(), 416, "YOLOX-Nano is a 416x416 network");

    // A blank frame should not crash, and every box must be inside the image.
    let blank = RgbImage::from_pixel(640, 360, Rgb([30, 32, 36]));
    let detections = detector.detect(&blank).unwrap();
    for d in &detections {
        assert!(d.bbox.x1 >= 0.0 && d.bbox.y1 >= 0.0, "{:?}", d.bbox);
        assert!(d.bbox.x2 <= 640.0 && d.bbox.y2 <= 360.0, "{:?}", d.bbox);
    }
}
