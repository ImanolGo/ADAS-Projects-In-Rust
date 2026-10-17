//! Object detection with YOLOX, run as ONNX by tract (pure Rust).
//!
//! YOLOX replaced the original YOLOv3 plan: the official Apache-2.0 ONNX
//! weights run directly under tract (the same runtime as the sign classifier),
//! so there is no Darknet engine to write and no Python to run. YOLOX-Nano is
//! ~3.7 MB, ~1.1 GFLOPs and ~25.8 mAP — faster *and* more accurate than
//! yolov3-tiny at a ninth of the size, and YOLOX-Tiny/S go further still.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use image::{imageops::FilterType, RgbImage};
use tract_onnx::prelude::*;

use crate::coco;

/// YOLOX's three detection strides: P3, P4, P5.
const STRIDES: [usize; 3] = [8, 16, 32];

/// The value YOLOX letterboxes with (its demo uses uint8 114).
const PAD: f32 = 114.0;

/// An axis-aligned rectangle in original-image pixels, `(x1, y1)` top-left.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BoundingBox {
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn intersection(&self, other: &Self) -> f32 {
        let w = (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0);
        let h = (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0);
        w * h
    }

    pub fn iou(&self, other: &Self) -> f32 {
        let union = self.area() + other.area() - self.intersection(other);
        if union <= 0.0 {
            0.0
        } else {
            self.intersection(other) / union
        }
    }
}

/// One detected object.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub class_id: usize,
    /// Objectness × class score, already through the sigmoid.
    pub confidence: f32,
    pub bbox: BoundingBox,
}

impl Detection {
    pub fn label(&self) -> &'static str {
        coco::label(self.class_id)
    }
}

/// Thresholds and class filter for [`ObjectDetector`].
#[derive(Clone, Debug)]
pub struct DetectorConfig {
    /// Minimum score to keep a box.
    pub confidence: f32,
    /// Same-class boxes overlapping more than this are suppressed.
    pub iou: f32,
    /// If set, only these COCO class ids are reported.
    pub keep: Option<Vec<usize>>,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self { confidence: 0.3, iou: 0.45, keep: None }
    }
}

impl DetectorConfig {
    /// Keep only vehicles, people and traffic signs.
    pub fn adas() -> Self {
        Self { keep: Some(coco::ADAS_CLASSES.to_vec()), ..Self::default() }
    }
}

/// A loaded YOLOX model.
pub struct ObjectDetector {
    plan: Arc<TypedRunnableModel>,
    /// Square network input side (416 for Nano/Tiny, 640 for S/M/L).
    size: usize,
    config: DetectorConfig,
}

impl ObjectDetector {
    /// Load with default thresholds.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with(path, DetectorConfig::default())
    }

    /// Load the ONNX model and read its square input size from the graph.
    pub fn load_with(path: impl AsRef<Path>, config: DetectorConfig) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            bail!(
                "detector model not found at {} (run scripts/fetch-data.sh to download YOLOX)",
                path.display()
            );
        }
        let model =
            tract_onnx::onnx().model_for_path(path).with_context(|| format!("reading {}", path.display()))?;
        let fact = model.input_fact(0)?;
        let dims = fact
            .shape
            .as_concrete_finite()?
            .with_context(|| format!("{} has a dynamic input shape", path.display()))?;
        let (height, width) = match dims.as_slice() {
            [1, 3, h, w] => (*h, *w),
            other => bail!("expected a 1x3xHxW input, got {other:?}"),
        };
        if height != width {
            bail!("expected a square input, got {height}x{width}");
        }
        if height % 32 != 0 {
            bail!("input side must be a multiple of 32, got {height}");
        }
        let plan = model.into_optimized()?.into_runnable()?;
        Ok(Self { plan, size: height, config })
    }

    /// The network's square input side.
    pub fn input_size(&self) -> usize {
        self.size
    }

    /// Replace the thresholds/filter.
    pub fn with_config(mut self, config: DetectorConfig) -> Self {
        self.config = config;
        self
    }

    /// Detect objects in one image, in original-image pixel coordinates.
    pub fn detect(&self, image: &RgbImage) -> Result<Vec<Detection>> {
        let (input, ratio) = self.preprocess(image);
        let tensor = tract_ndarray::Array4::from_shape_vec((1, 3, self.size, self.size), input)?;
        let outputs = self.plan.run(tvec!(Tensor::from(tensor).into()))?;
        let view = outputs[0].to_plain_array_view::<f32>()?;
        let data = view.as_slice().context("non-contiguous model output")?;
        let mut raw = Vec::new();
        self.decode(data, ratio, image.width(), image.height(), &mut raw)?;
        Ok(nms(raw, self.config.iou))
    }

    /// Letterbox to `size`x`size`, BGR, raw 0..=255, NCHW — exactly what the
    /// YOLOX ONNX export expects. Returns the buffer and the scale factor used
    /// to map boxes back.
    fn preprocess(&self, image: &RgbImage) -> (Vec<f32>, f32) {
        let size = self.size as f32;
        let ratio = (size / image.height() as f32).min(size / image.width() as f32);
        let nw = ((image.width() as f32 * ratio).round() as u32).clamp(1, self.size as u32);
        let nh = ((image.height() as f32 * ratio).round() as u32).clamp(1, self.size as u32);
        let resized = image::imageops::resize(image, nw, nh, FilterType::Triangle);

        let plane = self.size * self.size;
        let mut buf = vec![PAD; 3 * plane];
        for y in 0..nh as usize {
            for x in 0..nw as usize {
                let p = resized.get_pixel(x as u32, y as u32);
                let i = y * self.size + x;
                // OpenCV reads BGR and YOLOX never swaps it; match the reference.
                buf[i] = p[2] as f32;
                buf[plane + i] = p[1] as f32;
                buf[2 * plane + i] = p[0] as f32;
            }
        }
        (buf, ratio)
    }

    fn decode(
        &self,
        data: &[f32],
        ratio: f32,
        width: u32,
        height: u32,
        out: &mut Vec<Detection>,
    ) -> Result<()> {
        let stride_total = 5 + coco::NUM_CLASSES;
        let expected: usize = STRIDES.iter().map(|s| (self.size / s) * (self.size / s)).sum();
        if data.len() < expected * stride_total {
            bail!("model output has {} values, expected {}", data.len(), expected * stride_total);
        }
        let mut i = 0usize;
        for &stride in &STRIDES {
            let grid = self.size / stride;
            for y in 0..grid {
                for x in 0..grid {
                    let row = &data[i * stride_total..(i + 1) * stride_total];
                    i += 1;
                    // The exported head already applied sigmoid to objectness and
                    // class scores, so these are probabilities; only the box
                    // offsets/exp still need decoding.
                    let objectness = row[4];
                    if objectness < self.config.confidence {
                        continue;
                    }
                    let (class_id, score) =
                        row[5..].iter().enumerate().map(|(c, &prob)| (c, prob * objectness)).fold(
                            (0usize, f32::NEG_INFINITY),
                            |best, next| if next.1 > best.1 { next } else { best },
                        );
                    if score < self.config.confidence {
                        continue;
                    }
                    if let Some(keep) = &self.config.keep {
                        if !keep.contains(&class_id) {
                            continue;
                        }
                    }
                    let cx = (row[0] + x as f32) * stride as f32;
                    let cy = (row[1] + y as f32) * stride as f32;
                    let bw = row[2].exp() * stride as f32;
                    let bh = row[3].exp() * stride as f32;
                    let (w, h) = (width as f32, height as f32);
                    out.push(Detection {
                        class_id,
                        confidence: score,
                        bbox: BoundingBox {
                            x1: ((cx - bw / 2.0) / ratio).clamp(0.0, w),
                            y1: ((cy - bh / 2.0) / ratio).clamp(0.0, h),
                            x2: ((cx + bw / 2.0) / ratio).clamp(0.0, w),
                            y2: ((cy + bh / 2.0) / ratio).clamp(0.0, h),
                        },
                    });
                }
            }
        }
        Ok(())
    }
}

/// Greedy, per-class non-maximum suppression. Input need not be sorted.
pub fn nms(detections: Vec<Detection>, iou: f32) -> Vec<Detection> {
    let mut sorted = detections;
    sorted.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut kept: Vec<Detection> = Vec::new();
    for d in sorted {
        let clash = kept.iter().any(|k| k.class_id == d.class_id && k.bbox.iou(&d.bbox) > iou);
        if !clash {
            kept.push(d);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(class: usize, score: f32, x: f32) -> Detection {
        Detection {
            class_id: class,
            confidence: score,
            bbox: BoundingBox { x1: x, y1: 0.0, x2: x + 10.0, y2: 10.0 },
        }
    }

    #[test]
    fn iou_is_sane() {
        let a = BoundingBox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 };
        let b = BoundingBox { x1: 5.0, y1: 0.0, x2: 15.0, y2: 10.0 };
        assert!((a.iou(&a) - 1.0).abs() < 1e-6);
        assert!((a.iou(&b) - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(a.iou(&BoundingBox { x1: 20.0, y1: 0.0, x2: 30.0, y2: 10.0 }), 0.0);
    }

    #[test]
    fn nms_keeps_the_best_and_drops_overlaps_of_the_same_class() {
        let kept = nms(vec![det(2, 0.9, 0.0), det(2, 0.5, 1.0), det(2, 0.7, 100.0)], 0.45);
        let scores: Vec<f32> = kept.iter().map(|d| d.confidence).collect();
        assert_eq!(scores, vec![0.9, 0.7]);
    }

    #[test]
    fn nms_does_not_suppress_across_classes() {
        let kept = nms(vec![det(2, 0.9, 0.0), det(0, 0.8, 0.0)], 0.45);
        assert_eq!(kept.len(), 2);
    }
}
