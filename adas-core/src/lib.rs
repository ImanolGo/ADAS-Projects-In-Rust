//! Shared building blocks for the ADAS activities.
//!
//! - [`lanes`]: lane detection and lane-departure warning (pure Rust).
//! - [`classifier`]: the GTSRB traffic-sign CNN, run with tract (pure Rust ONNX).
//! - [`detector`]: YOLOX object detection, run with tract (pure Rust ONNX).
//! - [`coco`]: the 80 COCO class names and the driving-relevant subset.
//! - [`traffic_light`]: traffic-light colour recognition (RED/AMBER/GREEN/OFF).
//! - [`speed`]: debounced speed-limit tracking on top of the classifier.
//! - [`cluster`]: the instrument cluster renderer.
//! - [`render`]: drawing helpers and German road signs.
//! - [`overlay`]: drawing lanes and captions onto camera frames.
//! - [`video`]: video in/out through ffmpeg, or image folders.

pub mod classifier;
pub mod cluster;
pub mod coco;
pub mod detector;
pub mod labels;
pub mod lanes;
pub mod overlay;
pub mod paths;
pub mod render;
pub mod speed;
pub mod traffic_light;
pub mod video;

pub use classifier::{Prediction, SignClassifier};
pub use cluster::{ClusterState, LaneStatus};
pub use detector::{BoundingBox, Detection, DetectorConfig, ObjectDetector};
pub use lanes::{Departure, LaneConfig, LaneDetector, LaneFrame};
pub use speed::{LimitEvent, SpeedLimitTracker, TrackerConfig};
pub use traffic_light::{LightColor, LightReading, LightTracker};
