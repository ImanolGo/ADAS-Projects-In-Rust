//! Activity 5: traffic-light detection and colour recognition.
//!
//! ```text
//! cargo run --release -p street_light_detection                    # YOLOX if downloaded, else model-free
//! cargo run --release -p street_light_detection -- --preset tiny   # a more accurate YOLOX
//! cargo run --release -p street_light_detection -- --classical     # force the model-free detector
//! ```
//!
//! YOLOX locates traffic lights (COCO class 9) and
//! [`adas_core::traffic_light::classify`] reads the lamp colour from the crop.
//! If no model has been downloaded, a model-free detector finds bright
//! colourful blobs instead, so the activity still runs on a bare checkout.
//! Either way the colour is only *confirmed* after several frames agree.

use std::path::PathBuf;

use adas_core::paths;
use adas_core::render::{self, Fonts};
use adas_core::traffic_light::{self, LightCandidate, LightConfig};
use adas_core::video::{FrameSink, FrameSource};
use adas_core::{coco, overlay, DetectorConfig, LightColor, LightReading, LightTracker, ObjectDetector};
use anyhow::{Context, Result};
use clap::Parser;
use image::RgbImage;

#[derive(Parser)]
#[command(about = "Detect traffic lights and recognise their colour")]
struct Args {
    /// Image, folder of images, or video (default: data/video.mp4).
    #[arg(long)]
    input: Option<PathBuf>,
    /// Where to write the annotated result: .mp4, .webm, .gif, or a folder of PNGs.
    #[arg(long, default_value = "output/lights.mp4")]
    output: PathBuf,
    /// YOLOX ONNX model to locate the lights. Without it, a model-free detector is used.
    #[arg(long)]
    model: Option<PathBuf>,
    /// YOLOX preset: nano (fast), tiny, or s (accurate).
    #[arg(long, value_name = "PRESET")]
    preset: Option<String>,
    /// Force the model-free brightness/colour detector even if YOLOX is available.
    #[arg(long)]
    classical: bool,
    /// Minimum detector score.
    #[arg(long, default_value_t = 0.3)]
    conf: f32,
    /// How many frames must agree before the state changes.
    #[arg(long, default_value_t = 3)]
    confirm: usize,
    /// Start this many seconds into a video.
    #[arg(long)]
    start: Option<f32>,
    /// Only process this many seconds.
    #[arg(long)]
    duration: Option<f32>,
    /// Print one line per frame.
    #[arg(long)]
    verbose: bool,
}

fn detector_for(args: &Args) -> Result<Option<ObjectDetector>> {
    if args.classical {
        return Ok(None);
    }
    let config =
        DetectorConfig { confidence: args.conf, keep: Some(vec![coco::TRAFFIC_LIGHT]), ..Default::default() };
    let path = match (&args.model, &args.preset) {
        // No model asked for: use the default if it has been downloaded, else fall back.
        (None, None) => {
            let default = paths::default_detector();
            if default.exists() {
                default
            } else {
                return Ok(None);
            }
        }
        (Some(path), _) => path.clone(),
        (None, Some(name)) => {
            paths::detector_preset(name).with_context(|| format!("unknown preset '{name}'"))?
        }
    };
    Ok(Some(ObjectDetector::load_with(path, config)?))
}

/// Crop each YOLOX box and read its colour.
fn classify_boxes(frame: &RgbImage, detector: &ObjectDetector) -> Result<Vec<LightCandidate>> {
    let detections = detector.detect(frame)?;
    Ok(detections
        .into_iter()
        .map(|det| {
            let b = det.bbox;
            let x = b.x1.max(0.0) as u32;
            let y = b.y1.max(0.0) as u32;
            let w = (b.width().max(1.0) as u32).min(frame.width().saturating_sub(x)).max(1);
            let h = (b.height().max(1.0) as u32).min(frame.height().saturating_sub(y)).max(1);
            let crop = image::imageops::crop_imm(frame, x, y, w, h).to_image();
            LightCandidate { bbox: b, reading: traffic_light::classify(&crop) }
        })
        .collect())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let detector = detector_for(&args)?;
    let input = args.input.unwrap_or_else(paths::default_video);
    let source = FrameSource::open(&input, args.start, args.duration, 25.0)?;
    let info = source.info();
    let mut sink = FrameSink::create(&args.output, info.width, info.height, info.fps)?;
    let fonts = Fonts::load();
    let method = if detector.is_some() { "YOLOX boxes" } else { "model-free blobs" };

    let mut tracker = LightTracker::new(args.confirm);
    let (mut frames, mut changes, mut lit_frames) = (0usize, 0usize, 0usize);
    for frame in source {
        let frame = frame?;
        let lights = match &detector {
            Some(detector) => classify_boxes(&frame, detector)?,
            None => traffic_light::detect_lights(&frame, &LightConfig::default()),
        };
        frames += 1;
        let best = lights
            .iter()
            .filter(|l| l.reading.is_lit())
            .max_by(|a, b| a.reading.confidence.total_cmp(&b.reading.confidence));
        if best.is_some() {
            lit_frames += 1;
        }
        let reading =
            best.map(|l| l.reading).unwrap_or(LightReading { colour: LightColor::Unknown, confidence: 0.0 });
        if tracker.observe(reading) {
            changes += 1;
            println!("frame {frames}: traffic light → {}", tracker.current().as_str());
        }
        if args.verbose {
            for l in &lights {
                println!(
                    "frame {frames}: {:<6} {:.0}%  [{:.0},{:.0} {:.0}x{:.0}]",
                    l.reading.colour.as_str(),
                    l.reading.confidence * 100.0,
                    l.bbox.x1,
                    l.bbox.y1,
                    l.bbox.width(),
                    l.bbox.height()
                );
            }
        }

        let mut pix = render::pixmap_from_rgb(&frame);
        overlay::draw_lights(&mut pix, &fonts, &lights);
        overlay::light_panel(&mut pix, &fonts, tracker.current());
        sink.write(&render::rgb_from_pixmap(&pix))?;
    }
    sink.finish()?;

    println!(
        "{frames} frames, {method}. Lights visible in {lit_frames} frame(s); {} confirmed change(s); final state {}.",
        changes,
        tracker.current().as_str()
    );
    println!("Wrote {}", args.output.display());
    Ok(())
}
