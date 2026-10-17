//! Activity 2: traffic-sign recognition with the GTSRB CNN.
//!
//! ```text
//! cargo run --release -p traffic_sign_recognition -- data/Test            # Kaggle GTSRB layout
//! cargo run --release -p traffic_sign_recognition -- my_signs/ --gallery signs.png
//! ```
//!
//! With the Kaggle layout (`Test/*.png` next to `Test.csv`), the ground truth and
//! the sign's bounding box are read from the CSV, so you get a real accuracy
//! number instead of a list of guesses.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use adas_core::labels::label;
use adas_core::render::{self, rgba, Align, Fonts};
use adas_core::{paths, Prediction, SignClassifier};
use anyhow::{bail, Context, Result};
use clap::Parser;
use image::RgbImage;
use serde::Deserialize;

#[derive(Parser)]
#[command(about = "Classify traffic-sign images with the GTSRB model")]
struct Args {
    /// An image, or a folder of images (default: data/Test).
    input: Option<PathBuf>,
    /// GTSRB CSV with ClassId and Roi.* columns (default: <input>.csv next to the folder, if present).
    #[arg(long)]
    csv: Option<PathBuf>,
    /// ONNX model (default: data/Models/traffic-sign-recognition-model.onnx).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Only look at the first N images.
    #[arg(long)]
    limit: Option<usize>,
    /// Save a contact sheet of the first 48 results to this PNG.
    #[arg(long)]
    gallery: Option<PathBuf>,
    /// Print one line per image.
    #[arg(long)]
    verbose: bool,
}

#[derive(Debug, Deserialize)]
struct Row {
    #[serde(rename = "Roi.X1")]
    x1: u32,
    #[serde(rename = "Roi.Y1")]
    y1: u32,
    #[serde(rename = "Roi.X2")]
    x2: u32,
    #[serde(rename = "Roi.Y2")]
    y2: u32,
    #[serde(rename = "ClassId")]
    class_id: usize,
    #[serde(rename = "Path")]
    path: String,
}

struct Sample {
    path: PathBuf,
    truth: Option<usize>,
    roi: Option<(u32, u32, u32, u32)>,
}

fn samples(input: &Path, csv: Option<PathBuf>) -> Result<Vec<Sample>> {
    if input.is_file() {
        return Ok(vec![Sample { path: input.to_path_buf(), truth: None, roi: None }]);
    }
    if !input.is_dir() {
        bail!(
            "{} not found. Put GTSRB's Test/ and Test.csv in data/ (see README), or pass a folder of images.",
            input.display()
        );
    }
    let csv = csv.or_else(|| Some(input.with_extension("csv")).filter(|p| p.exists()));
    if let Some(csv) = csv {
        // Kaggle's Test.csv paths look like "Test/00000.png", relative to the CSV's folder.
        let base = csv.parent().unwrap_or(Path::new("."));
        let mut reader =
            csv::Reader::from_path(&csv).with_context(|| format!("reading {}", csv.display()))?;
        return reader
            .deserialize::<Row>()
            .map(|row| {
                let row = row?;
                Ok(Sample {
                    path: base.join(&row.path),
                    truth: Some(row.class_id),
                    roi: Some((row.x1, row.y1, row.x2, row.y2)),
                })
            })
            .collect();
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(input)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| ["png", "jpg", "jpeg", "ppm"].contains(&e.to_ascii_lowercase().as_str()))
        })
        .collect();
    paths.sort();
    Ok(paths.into_iter().map(|path| Sample { path, truth: None, roi: None }).collect())
}

fn load(sample: &Sample) -> Result<RgbImage> {
    let img =
        image::open(&sample.path).with_context(|| format!("reading {}", sample.path.display()))?.to_rgb8();
    Ok(match sample.roi {
        // Crop to the annotated sign, as the network was trained on.
        Some((x1, y1, x2, y2)) if x2 > x1 && y2 > y1 && x2 <= img.width() && y2 <= img.height() => {
            image::imageops::crop_imm(&img, x1, y1, x2 - x1, y2 - y1).to_image()
        }
        _ => img,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input = args.input.clone().unwrap_or_else(paths::default_test_dir);
    let model = SignClassifier::load(args.model.clone().unwrap_or_else(paths::default_model))?;
    let mut all = samples(&input, args.csv.clone())?;
    if let Some(n) = args.limit {
        all.truncate(n);
    }
    if all.is_empty() {
        bail!("no images found in {}", input.display());
    }

    let started = Instant::now();
    let mut results: Vec<(Sample, RgbImage, Prediction)> = Vec::new();
    for sample in all {
        let img = load(&sample)?;
        let prediction = model.classify(&img)?;
        if args.verbose || sample.truth.is_none() {
            let mark = match sample.truth {
                Some(t) if t == prediction.class_id => "✓ ",
                Some(_) => "✗ ",
                None => "",
            };
            println!(
                "{mark}{}: {} ({:.0}%)",
                sample.path.file_name().unwrap_or_default().to_string_lossy(),
                prediction.label(),
                prediction.confidence * 100.0
            );
        }
        results.push((sample, img, prediction));
    }
    let elapsed = started.elapsed().as_secs_f32();
    println!(
        "\nClassified {} images in {elapsed:.2}s ({:.1} ms each).",
        results.len(),
        1000.0 * elapsed / results.len() as f32
    );

    let labelled: Vec<&(Sample, RgbImage, Prediction)> =
        results.iter().filter(|r| r.0.truth.is_some()).collect();
    if !labelled.is_empty() {
        report_accuracy(&labelled);
    }
    if let Some(path) = &args.gallery {
        gallery(&results, path)?;
        println!("Wrote {}", path.display());
    }
    Ok(())
}

fn report_accuracy(labelled: &[&(Sample, RgbImage, Prediction)]) {
    let correct = labelled.iter().filter(|r| r.0.truth == Some(r.2.class_id)).count();
    println!(
        "Accuracy: {correct}/{} = {:.2}%",
        labelled.len(),
        100.0 * correct as f32 / labelled.len() as f32
    );

    let mut per_class: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut confusions: HashMap<(usize, usize), usize> = HashMap::new();
    for (sample, _, p) in labelled {
        let truth = sample.truth.unwrap();
        let entry = per_class.entry(truth).or_default();
        entry.1 += 1;
        if truth == p.class_id {
            entry.0 += 1;
        } else {
            *confusions.entry((truth, p.class_id)).or_default() += 1;
        }
    }
    let mut worst: Vec<_> = per_class.into_iter().collect();
    worst.sort_by(|a, b| (a.1 .0 as f32 / a.1 .1 as f32).total_cmp(&(b.1 .0 as f32 / b.1 .1 as f32)));
    println!("\nHardest classes:");
    for (class, (ok, n)) in worst.iter().take(5) {
        println!("  {:5.1}%  {} ({ok}/{n})", 100.0 * *ok as f32 / *n as f32, label(*class));
    }
    let mut top: Vec<_> = confusions.into_iter().collect();
    top.sort_by_key(|t| std::cmp::Reverse(t.1));
    println!("\nMost common mistakes:");
    for ((truth, said), n) in top.iter().take(5) {
        println!("  {n:4}×  {} → {}", label(*truth), label(*said));
    }
}

/// A contact sheet: each crop with the model's answer, framed green (right) or red (wrong).
fn gallery(results: &[(Sample, RgbImage, Prediction)], path: &Path) -> Result<()> {
    let fonts = Fonts::load();
    let (cols, cell, caption) = (8u32, 150u32, 46u32);
    let shown = &results[..results.len().min(48)];
    let rows = (shown.len() as u32).div_ceil(cols);
    let mut sheet = tiny_skia::Pixmap::new(cols * cell, rows * (cell + caption)).context("sheet size")?;
    sheet.fill(rgba(20, 22, 26, 1.0));
    for (i, (sample, img, p)) in shown.iter().enumerate() {
        let (cx, cy) = ((i as u32 % cols) * cell, (i as u32 / cols) * (cell + caption));
        let thumb = image::imageops::resize(img, cell - 16, cell - 16, image::imageops::FilterType::Triangle);
        let frame = match sample.truth {
            Some(t) if t == p.class_id => rgba(64, 210, 120, 1.0),
            Some(_) => rgba(255, 69, 58, 1.0),
            None => rgba(60, 64, 70, 1.0),
        };
        render::fill_rect(
            &mut sheet,
            cx as f32 + 4.0,
            cy as f32 + 4.0,
            cell as f32 - 8.0,
            cell as f32 - 8.0,
            frame,
        );
        let thumb_pix = render::pixmap_from_rgb(&thumb);
        sheet.draw_pixmap(
            cx as i32 + 8,
            cy as i32 + 8,
            thumb_pix.as_ref(),
            &tiny_skia::PixmapPaint::default(),
            tiny_skia::Transform::identity(),
            None,
        );
        let name = p.label().replace("Speed limit ", "").replace(['(', ')'], "");
        let text = if name.chars().count() > 18 {
            format!("{}…", name.chars().take(17).collect::<String>())
        } else {
            name
        };
        let x = cx as f32 + cell as f32 / 2.0;
        render::text(
            &mut sheet,
            &fonts.semibold,
            &text,
            x,
            (cy + cell + 16) as f32,
            15.0,
            rgba(238, 241, 243, 1.0),
            Align::Center,
        );
        let conf = format!("{:.0}%", p.confidence * 100.0);
        render::text(
            &mut sheet,
            &fonts.semibold,
            &conf,
            x,
            (cy + cell + 36) as f32,
            13.0,
            rgba(138, 146, 156, 1.0),
            Align::Center,
        );
    }
    render::rgb_from_pixmap(&sheet).save(path)?;
    Ok(())
}
