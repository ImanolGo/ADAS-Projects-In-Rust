//! Activity 1: lane detection and lane-departure warning on a dashcam video.
//!
//! ```text
//! cargo run --release -p lane_detection -- --output output/lanes.mp4
//! cargo run --release -p lane_detection -- --input data/challenge.mp4 --output lanes.gif --debug
//! ```

use std::path::PathBuf;
use std::time::Instant;

use adas_core::lanes::{Departure, LaneDetector};
use adas_core::overlay;
use adas_core::paths;
use adas_core::render::{self, Fonts};
use adas_core::video::{FrameSink, FrameSource};
use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(about = "Detect lanes and warn about lane departure in a dashcam video")]
struct Args {
    /// Video file or folder of frames (default: data/video.mp4).
    #[arg(long)]
    input: Option<PathBuf>,
    /// Where to write the annotated result: .mp4, .webm, .gif, or a folder of PNGs.
    #[arg(long, default_value = "output/lanes.mp4")]
    output: PathBuf,
    /// Start this many seconds into the video.
    #[arg(long)]
    start: Option<f32>,
    /// Only process this many seconds.
    #[arg(long)]
    duration: Option<f32>,
    /// Show the edge map the detector works on, top right.
    #[arg(long)]
    debug: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input = args.input.unwrap_or_else(paths::default_video);
    let source = FrameSource::open(&input, args.start, args.duration, 25.0)?;
    let info = source.info();
    let mut sink = FrameSink::create(&args.output, info.width, info.height, info.fps)?;
    let mut detector = LaneDetector::default();
    let fonts = Fonts::load();

    let (mut frames, mut both, mut departures) = (0usize, 0usize, 0usize);
    let started = Instant::now();
    for frame in source {
        let mut frame = frame?;
        let lanes = detector.detect(&frame);
        frames += 1;
        both += usize::from(lanes.left.is_some() && lanes.right.is_some());
        departures += usize::from(lanes.departure != Departure::None);

        if args.debug {
            let scale = (frame.width() as f32 / detector.config.work_width as f32).max(1.0);
            let small = image::imageops::resize(
                &frame,
                (frame.width() as f32 / scale) as u32,
                (frame.height() as f32 / scale) as u32,
                image::imageops::FilterType::Triangle,
            );
            overlay::edges_inset(&mut frame, &detector.roi_edges(&small));
        }
        let mut pix = render::pixmap_from_rgb(&frame);
        overlay::draw_lanes(&mut pix, &lanes);
        let status = match (lanes.departure, lanes.offset) {
            (Departure::Left, _) => "Drifting left".to_string(),
            (Departure::Right, _) => "Drifting right".to_string(),
            (Departure::None, Some(o)) => format!("Centred · offset {:+.0}% of lane", o * 100.0),
            (Departure::None, None) => "Looking for lanes".to_string(),
        };
        overlay::caption(&mut pix, &fonts, &status);
        sink.write(&render::rgb_from_pixmap(&pix))?;
    }
    sink.finish()?;

    let secs = started.elapsed().as_secs_f32();
    println!(
        "{frames} frames in {secs:.1}s ({:.0} fps). Both lanes found in {:.0}% of frames, departure warnings in {departures}.",
        frames as f32 / secs.max(1e-3),
        100.0 * both as f32 / frames.max(1) as f32
    );
    println!("Wrote {}", args.output.display());
    Ok(())
}
