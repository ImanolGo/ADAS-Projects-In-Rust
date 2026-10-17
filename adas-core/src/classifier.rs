//! Traffic-sign classification with the bundled GTSRB CNN (ONNX, run by tract).

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use image::{imageops::FilterType, RgbImage};
use tract_onnx::prelude::*;

use crate::labels::{label, speed_meaning, SpeedMeaning};

/// The network's input size (GTSRB crops resized to 30x30, NHWC).
pub const INPUT_SIZE: u32 = 30;
pub const NUM_CLASSES: usize = 43;

/// One classification.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub class_id: usize,
    /// Softmax probability of `class_id`.
    ///
    /// Careful: this network is very overconfident on images unlike its
    /// training data (it reports ~1.0 even when wrong), which is why speed
    /// limits are confirmed over several frames instead of trusted per frame.
    pub confidence: f32,
    /// The three most likely classes, best first.
    pub top3: [(usize, f32); 3],
}

impl Prediction {
    pub fn label(&self) -> &'static str {
        label(self.class_id)
    }

    pub fn speed_meaning(&self) -> Option<SpeedMeaning> {
        speed_meaning(self.class_id)
    }
}

/// Turn probabilities into a [`Prediction`]. NaN-safe (the original
/// `partial_cmp(..).unwrap()` would panic on a NaN output).
pub fn prediction_from_scores(scores: &[f32]) -> Result<Prediction> {
    if scores.len() != NUM_CLASSES {
        bail!("expected {NUM_CLASSES} scores, got {}", scores.len());
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    let key = |i: usize| if scores[i].is_nan() { f32::NEG_INFINITY } else { scores[i] };
    order.sort_by(|&a, &b| key(b).total_cmp(&key(a)));
    let top3 = [(order[0], scores[order[0]]), (order[1], scores[order[1]]), (order[2], scores[order[2]])];
    Ok(Prediction { class_id: order[0], confidence: scores[order[0]], top3 })
}

/// Resize to 30x30 (bilinear) and lay out as NHWC floats in 0..=1.
///
/// On the 51 bundled GTSRB test images, cropped to their annotated box, this
/// gets 50/51; the original code (whole image, nearest-neighbour) got 48/51.
/// Feeding raw 0..255 instead of `/255` made no reliable difference.
pub fn preprocess(image: &RgbImage) -> Vec<f32> {
    let resized = image::imageops::resize(image, INPUT_SIZE, INPUT_SIZE, FilterType::Triangle);
    resized.as_raw().iter().map(|&v| v as f32 / 255.0).collect()
}

/// The GTSRB classifier.
pub struct SignClassifier {
    plan: Arc<TypedRunnableModel>,
}

impl SignClassifier {
    /// Load and optimise the ONNX model once. Inference is then ~1 ms per crop on a CPU.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            bail!("model not found at {} (set ADAS_DATA_DIR or pass --model)", path.display());
        }
        let side = INPUT_SIZE as usize;
        let plan = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("reading {}", path.display()))?
            .with_input_fact(0, f32::fact([1, side, side, 3]).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { plan })
    }

    /// Classify one sign crop (any size; it is resized to 30x30).
    pub fn classify(&self, crop: &RgbImage) -> Result<Prediction> {
        let side = INPUT_SIZE as usize;
        let input = tract_ndarray::Array4::from_shape_vec((1, side, side, 3), preprocess(crop))?;
        let outputs = self.plan.run(tvec!(Tensor::from(input).into()))?;
        let scores = outputs[0].to_plain_array_view::<f32>()?;
        prediction_from_scores(scores.as_slice().context("non-contiguous model output")?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top3_is_sorted_and_nan_safe() {
        let mut scores = vec![0.0f32; NUM_CLASSES];
        scores[4] = 0.7;
        scores[1] = 0.2;
        scores[7] = 0.1;
        scores[9] = f32::NAN;
        let p = prediction_from_scores(&scores).unwrap();
        assert_eq!(p.class_id, 4);
        assert_eq!(p.top3.map(|t| t.0), [4, 1, 7]);
        assert_eq!(p.speed_meaning(), Some(SpeedMeaning::Limit(70)));
        assert!(prediction_from_scores(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn preprocess_is_nhwc_30x30_unit_range() {
        let img = RgbImage::from_pixel(64, 48, image::Rgb([0, 51, 255]));
        let v = preprocess(&img);
        assert_eq!(v.len(), 30 * 30 * 3);
        assert_eq!(&v[..3], &[0.0, 0.2, 1.0]);
    }
}
