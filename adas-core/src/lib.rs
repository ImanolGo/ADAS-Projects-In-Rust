//! Shared building blocks for the ADAS activities.
//!
//! - [`lanes`]: lane detection and lane-departure warning (pure Rust).
//! - [`classifier`]: the GTSRB traffic-sign CNN, run with tract (pure Rust ONNX).
//! - [`speed`]: debounced speed-limit tracking on top of the classifier.
//! - [`cluster`]: the instrument cluster renderer.
//! - [`render`]: drawing helpers and German road signs.
//! - [`overlay`]: drawing lanes and captions onto camera frames.
//! - [`video`]: video in/out through ffmpeg, or image folders.

pub mod classifier;
pub mod cluster;
pub mod labels;
pub mod lanes;
pub mod overlay;
pub mod paths;
pub mod render;
pub mod speed;
pub mod video;

pub use classifier::{Prediction, SignClassifier};
pub use cluster::{ClusterState, LaneStatus};
pub use lanes::{Departure, LaneConfig, LaneDetector, LaneFrame};
pub use speed::{LimitEvent, SpeedLimitTracker, TrackerConfig};
