//! End-to-end: the bundled ONNX model, run by tract, on signs drawn by our renderer.

use adas_core::labels::{class_for_limit, speed_meaning, SpeedMeaning};
use adas_core::render::{self, Fonts, CROP_PER_RADIUS};
use adas_core::{paths, SignClassifier};
use image::RgbImage;
use tiny_skia::Pixmap;

/// A sign crop framed the way GTSRB crops are (sign ≈ 72% of the side).
fn crop(
    fonts: &Fonts,
    background: (u8, u8, u8),
    draw: impl Fn(&mut Pixmap, &Fonts, f32, f32, f32),
) -> RgbImage {
    let side = 96.0;
    let mut pix = Pixmap::new(side as u32, side as u32).unwrap();
    pix.fill(render::rgba(background.0, background.1, background.2, 1.0));
    let r = side / CROP_PER_RADIUS;
    draw(&mut pix, fonts, side / 2.0, side / 2.0, r);
    render::rgb_from_pixmap(&pix)
}

#[test]
fn model_loads_and_reads_the_signs_it_is_good_at() {
    let model = SignClassifier::load(paths::default_model()).expect("bundled model loads");
    let fonts = Fonts::load();
    let backgrounds = [(120, 140, 110), (150, 170, 200), (90, 90, 90)];
    for kmh in [20, 30, 80] {
        for bg in backgrounds {
            let img = crop(&fonts, bg, |p, f, x, y, r| render::speed_sign(p, f, x, y, r, kmh));
            let prediction = model.classify(&img).unwrap();
            assert_eq!(
                prediction.class_id,
                class_for_limit(kmh).unwrap(),
                "{kmh} km/h on {bg:?} read as {} ({:.2})",
                prediction.label(),
                prediction.confidence
            );
        }
    }
}

#[test]
fn every_prediction_is_a_valid_class() {
    let model = SignClassifier::load(paths::default_model()).unwrap();
    let noise =
        RgbImage::from_fn(41, 37, |x, y| image::Rgb([(x * 7) as u8, (y * 11) as u8, ((x ^ y) * 13) as u8]));
    let p = model.classify(&noise).unwrap();
    assert!(p.class_id < 43 && (0.0..=1.0001).contains(&p.confidence));
    // Whatever it says, the speed meaning must agree with the label table.
    if let Some(SpeedMeaning::Limit(kmh)) = speed_meaning(p.class_id) {
        assert!(p.label().contains(&kmh.to_string()));
    }
}

#[test]
fn a_missing_model_is_a_clear_error() {
    let err = SignClassifier::load("/no/such/model.onnx").err().unwrap();
    assert!(err.to_string().contains("ADAS_DATA_DIR"));
}
