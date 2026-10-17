//! How well does the classifier read the signs we render, at the sizes they
//! appear on screen? `cargo run -p adas-core --example sign_probe`
//!
//! A quick way to see the model's blind spots: it was trained on GTSRB photos
//! and generalises unevenly to clean, synthetic signs.
use adas_core::labels::class_for_limit;
use adas_core::render::{self, Fonts, CROP_PER_RADIUS};
use adas_core::{paths, SignClassifier};
use tiny_skia::Pixmap;

fn main() -> anyhow::Result<()> {
    let model = SignClassifier::load(paths::default_model())?;
    let fonts = Fonts::load();
    let backgrounds = [(120, 140, 110), (150, 170, 200), (90, 90, 90), (170, 160, 140), (60, 70, 60)];
    for kmh in [20, 30, 50, 60, 70, 80, 100, 120] {
        let (mut ok, mut n) = (0, 0);
        for bg in backgrounds {
            for r in [8.0f32, 11.0, 15.0, 20.0, 28.0, 36.0] {
                let side = (r * CROP_PER_RADIUS).round().max(4.0);
                let mut pix = Pixmap::new(side as u32, side as u32).unwrap();
                pix.fill(render::rgba(bg.0, bg.1, bg.2, 1.0));
                render::speed_sign(&mut pix, &fonts, side / 2.0, side / 2.0, r, kmh);
                let p = model.classify(&render::rgb_from_pixmap(&pix))?;
                ok += (p.class_id == class_for_limit(kmh).unwrap()) as u32;
                n += 1;
            }
        }
        println!("{kmh:>4} km/h: {ok}/{n}");
    }
    Ok(())
}
