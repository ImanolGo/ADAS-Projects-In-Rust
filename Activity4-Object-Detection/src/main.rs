//! Activity 4: object detection with YOLOX.
//!
//! ```text
//! cargo run --release -p object_detection                                  # road clip → output/detect.mp4
//! cargo run --release -p object_detection -- --input street.jpg --gallery out.png
//! cargo run --release -p object_detection -- --preset tiny --classes all
//! ```
//!
//! YOLOX replaced the original YOLOv3 plan: the official Apache-2.0 ONNX weights
//! run under tract, the same pure-Rust runtime as the sign classifier, so there
//! is no Darknet engine and no Python. `--preset nano` (default) is fast enough
//! for video; `tiny` and `s` trade speed for accuracy.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use adas_core::coco;
use adas_core::detector::{DetectorConfig, ObjectDetector};
use adas_core::overlay;
use adas_core::paths;
use adas_core::render::{self, rgba, Fonts};
use adas_core::video::{FrameSink, FrameSource};
use anyhow::{bail, Context, Result};
use clap::Parser;

#[derive(Parser)]
#[command(about = "Detect objects with YOLOX (ONNX + tract, pure Rust)")]
struct Args {
    /// Image, folder of images, or video (default: data/video.mp4).
    #[arg(long)]
    input: Option<PathBuf>,
    /// Where to write the annotated result: .mp4, .webm, .gif, or a folder of PNGs.
    #[arg(long, default_value = "output/detect.mp4")]
    output: PathBuf,
    /// YOLOX ONNX model.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Model preset: nano (fast), tiny, or s (accurate).
    #[arg(long, value_name = "PRESET")]
    preset: Option<String>,
    /// Minimum score to keep a box.
    #[arg(long, default_value_t = 0.3)]
    conf: f32,
    /// NMS IoU threshold.
    #[arg(long, default_value_t = 0.45)]
    iou: f32,
    /// Which classes to keep: "adas", "all", or a comma-separated list of names.
    #[arg(long, default_value = "adas")]
    classes: String,
    /// Start this many seconds into a video.
    #[arg(long)]
    start: Option<f32>,
    /// Only process this many seconds.
    #[arg(long)]
    duration: Option<f32>,
    /// Save a contact sheet of annotated images to this PNG.
    #[arg(long)]
    gallery: Option<PathBuf>,
    /// Print one line per detected object.
    #[arg(long)]
    verbose: bool,
}

fn parse_classes(spec: &str) -> Result<Option<Vec<usize>>> {
    match spec.trim() {
        "all" | "" => Ok(None),
        "adas" => Ok(Some(coco::ADAS_CLASSES.to_vec())),
        list => list
            .split(',')
            .map(|name| {
                let name = name.trim().to_ascii_lowercase();
                name.parse::<usize>()
                    .ok()
                    .or_else(|| coco::id(&name))
                    .with_context(|| format!("unknown class '{name}'"))
            })
            .collect::<Result<Vec<_>>>()
            .map(Some),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model = match (&args.model, &args.preset) {
        (Some(path), _) => path.clone(),
        (None, Some(name)) => {
            paths::detector_preset(name).with_context(|| format!("unknown preset '{name}'"))?
        }
        (None, None) => paths::default_detector(),
    };
    let classes = parse_classes(&args.classes)?;
    let config = DetectorConfig { confidence: args.conf, iou: args.iou, keep: classes.clone() };
    let detector = ObjectDetector::load_with(&model, config)?;

    let input = args.input.unwrap_or_else(paths::default_video);
    let source = FrameSource::open(&input, args.start, args.duration, 25.0)?;
    let info = source.info();
    let mut sink = FrameSink::create(&args.output, info.width, info.height, info.fps)?;
    let fonts = Fonts::load();
    let want_gallery = args.gallery.is_some();
    let extended = input.extension().is_some();

    let mut counts: HashMap<usize, usize> = HashMap::new();
    let mut gallery: Vec<image::RgbImage> = Vec::new();
    let (mut frames, mut objects) = (0usize, 0usize);
    let started = Instant::now();
    for frame in source {
        let frame = frame?;
        let detections = detector.detect(&frame)?;
        frames += 1;
        objects += detections.len();
        for d in &detections {
            *counts.entry(d.class_id).or_default() += 1;
        }
        if args.verbose {
            if detections.is_empty() {
                println!("frame {frames}: nothing");
            }
            for d in &detections {
                println!(
                    "frame {frames}: {:<14} {:.0}%  [{:.0},{:.0} {:.0}x{:.0}]",
                    d.label(),
                    d.confidence * 100.0,
                    d.bbox.x1,
                    d.bbox.y1,
                    d.bbox.width(),
                    d.bbox.height()
                );
            }
        }
        let mut pix = render::pixmap_from_rgb(&frame);
        overlay::draw_detections(&mut pix, &fonts, &detections);
        let n = detections.len();
        overlay::caption(&mut pix, &fonts, &format!("{n} object{}", if n == 1 { "" } else { "s" }));
        let annotated = render::rgb_from_pixmap(&pix);
        sink.write(&annotated)?;
        if want_gallery && gallery.len() < 12 {
            gallery.push(annotated);
        }
    }
    sink.finish()?;

    let secs = started.elapsed().as_secs_f32();
    println!(
        "{frames} frames in {secs:.1}s ({:.1} fps with {} input). {} objects total.",
        frames as f32 / secs.max(1e-3),
        if extended { "video" } else { "images" },
        objects
    );
    let mut ranked: Vec<_> = counts.into_iter().collect();
    ranked.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    if !ranked.is_empty() {
        print!("Detected: ");
        let text: Vec<String> = ranked.iter().map(|&(c, n)| format!("{} ×{n}", coco::label(c))).collect();
        println!("{}", text.join(", "));
    }
    if classes.is_some() {
        println!("(filtered to {})", args.classes);
    }
    if let Some(path) = &args.gallery {
        contact_sheet(&gallery, path)?;
        println!("Wrote {}", path.display());
    }
    println!("Wrote {}", args.output.display());
    Ok(())
}

/// A grid of annotated frames, best-effort: what the detector saw.
fn contact_sheet(frames: &[image::RgbImage], path: &Path) -> Result<()> {
    if frames.is_empty() {
        bail!("no frames to put in the contact sheet");
    }
    let cols = 3u32;
    let cell_w = 360u32;
    let first = &frames[0];
    let cell_h = (first.height() * cell_w / first.width().max(1)).max(1);
    let rows = (frames.len() as u32).div_ceil(cols);
    let mut sheet = tiny_skia::Pixmap::new(cols * cell_w, rows * cell_h).context("sheet size")?;
    sheet.fill(rgba(16, 18, 22, 1.0));
    for (i, frame) in frames.iter().enumerate() {
        let thumb = image::imageops::resize(frame, cell_w, cell_h, image::imageops::FilterType::Triangle);
        let pix = render::pixmap_from_rgb(&thumb);
        sheet.draw_pixmap(
            ((i as u32 % cols) * cell_w) as i32,
            ((i as u32 / cols) * cell_h) as i32,
            pix.as_ref(),
            &tiny_skia::PixmapPaint::default(),
            tiny_skia::Transform::identity(),
            None,
        );
    }
    render::rgb_from_pixmap(&sheet).save(path)?;
    Ok(())
}
