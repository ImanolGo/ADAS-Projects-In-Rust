//! Activity 3: speed-limit recognition over time.
//!
//! The classifier answers per image; a car needs *the limit in force*. This
//! binary feeds a sequence of frames through the classifier and a
//! [`SpeedLimitTracker`], which only changes the limit when several recent
//! frames agree, and which understands "end of limit" signs.
//!
//! ```text
//! cargo run --release -p speed_limit_recognition -- --demo        # no data needed
//! cargo run --release -p speed_limit_recognition -- frames/       # your own crops, in order
//! ```

use std::path::PathBuf;

use adas_core::render::{self, rgba, Fonts, CROP_PER_RADIUS};
use adas_core::speed::{LimitEvent, SpeedLimitTracker, TrackerConfig};
use adas_core::{paths, Prediction, SignClassifier};
use anyhow::{bail, Result};
use clap::Parser;
use image::RgbImage;

#[derive(Parser)]
#[command(about = "Track the speed limit in force from a sequence of sign images")]
struct Args {
    /// Folder of sign crops, read in file-name order as consecutive frames.
    input: Option<PathBuf>,
    /// Run on a synthetic drive past rendered signs instead (no data needed).
    #[arg(long)]
    demo: bool,
    #[arg(long)]
    model: Option<PathBuf>,
    /// Ignore predictions below this probability.
    #[arg(long, default_value_t = 0.6)]
    min_confidence: f32,
    /// Frames to vote over.
    #[arg(long, default_value_t = 9)]
    window: usize,
    /// Votes needed to change the limit.
    #[arg(long, default_value_t = 5)]
    votes: usize,
    /// The winning reading must lead the runner-up by this many votes.
    #[arg(long, default_value_t = 2)]
    margin: usize,
}

/// One frame of the drive: a sign crop, or nothing in view.
enum Frame {
    Sign(RgbImage),
    Empty,
}

/// A drive past signs, each seen for a few frames as it grows closer.
/// The 70 is one the model often misreads, so voting has work to do.
fn demo_frames() -> Vec<(String, Frame)> {
    let fonts = Fonts::load();
    let backgrounds = [(118, 138, 112), (150, 168, 196), (96, 98, 96)];
    let mut frames = Vec::new();
    let gap = |frames: &mut Vec<(String, Frame)>, n: usize| {
        frames.extend((0..n).map(|_| ("—".to_string(), Frame::Empty)));
    };
    for (i, (name, kmh, end)) in
        [("30 zone", 30, false), ("70 ahead", 70, false), ("80 ahead", 80, false), ("end of 80", 80, true)]
            .into_iter()
            .enumerate()
    {
        gap(&mut frames, 3);
        for step in 0..10 {
            let r = 9.0 + step as f32 * 2.5; // the sign grows as we approach
            let side = (r * CROP_PER_RADIUS).round();
            let bg = backgrounds[(i + step) % backgrounds.len()];
            let mut pix = tiny_skia::Pixmap::new(side as u32, side as u32).unwrap();
            pix.fill(rgba(bg.0, bg.1, bg.2, 1.0));
            if end {
                render::end_of_limit_sign(&mut pix, &fonts, side / 2.0, side / 2.0, r, kmh);
            } else {
                render::speed_sign(&mut pix, &fonts, side / 2.0, side / 2.0, r, kmh);
            }
            frames.push((name.to_string(), Frame::Sign(render::rgb_from_pixmap(&pix))));
        }
    }
    gap(&mut frames, 3);
    frames
}

fn folder_frames(dir: &PathBuf) -> Result<Vec<(String, Frame)>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| {
                ["png", "jpg", "jpeg", "ppm"].contains(&e.to_string_lossy().to_lowercase().as_str())
            })
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("no images in {}", dir.display());
    }
    paths
        .into_iter()
        .map(|p| {
            Ok((
                p.file_name().unwrap_or_default().to_string_lossy().into_owned(),
                Frame::Sign(image::open(&p)?.to_rgb8()),
            ))
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let frames = match (&args.input, args.demo) {
        (_, true) => demo_frames(),
        (Some(dir), false) => folder_frames(dir)?,
        (None, false) => bail!("pass a folder of sign images, or --demo"),
    };
    let model = SignClassifier::load(args.model.clone().unwrap_or_else(paths::default_model))?;
    let mut tracker = SpeedLimitTracker::new(TrackerConfig {
        min_confidence: args.min_confidence,
        window: args.window,
        votes: args.votes,
        margin: args.margin,
    });

    println!("{:>5}  {:<14} {:<30} {:>5}  limit", "frame", "scene", "camera reads", "conf");
    let mut events = 0;
    // Per sighting (demo): (scene, frames, correct reads, limit when it ended).
    let mut sightings: Vec<(String, usize, usize, Option<u32>)> = Vec::new();
    for (i, (scene, frame)) in frames.iter().enumerate() {
        let prediction: Option<Prediction> = match frame {
            Frame::Sign(img) => Some(model.classify(img)?),
            Frame::Empty => None,
        };
        let event = tracker.observe(prediction.as_ref());
        let (read, conf) = match &prediction {
            Some(p) => (p.label().to_string(), format!("{:.0}%", p.confidence * 100.0)),
            None => ("no sign in view".into(), String::new()),
        };
        if let (Some(p), true) = (&prediction, args.demo) {
            let expected = scene.split_whitespace().find_map(|w| w.parse::<u32>().ok());
            let correct = if scene.starts_with("end") {
                p.class_id == 6
            } else {
                expected.is_some_and(|k| p.label().contains(&format!("({k}km/h)")))
            };
            if sightings.last().is_none_or(|s| &s.0 != scene) {
                sightings.push((scene.clone(), 0, 0, None));
            }
            let current = sightings.last_mut().unwrap();
            current.1 += 1;
            current.2 += usize::from(correct);
        }
        let limit = tracker.current().map_or("–".into(), |k| k.to_string());
        let note = match event {
            Some(LimitEvent::Set(k)) => format!("   ← new limit {k} km/h"),
            Some(LimitEvent::Cleared) => "   ← limit lifted".into(),
            None => String::new(),
        };
        events += usize::from(event.is_some());
        if let Some(last) = sightings.last_mut() {
            last.3 = tracker.current();
        }
        println!("{i:>5}  {scene:<14} {read:<30} {conf:>5}  {limit}{note}");
    }
    println!("\n{events} limit changes from {} frames.", frames.len());
    if args.demo {
        println!("\nHow each sign went:");
        for (scene, seen, correct, limit) in &sightings {
            let expected = if scene.starts_with("end") {
                None
            } else {
                scene.split_whitespace().find_map(|w| w.parse::<u32>().ok())
            };
            let verdict = if *limit == expected { "✓" } else { "✗" };
            let shown = limit.map_or("none".to_string(), |k| format!("{k} km/h"));
            println!("  {verdict} {scene:<10} read correctly in {correct}/{seen} frames → limit {shown}");
        }
        println!("\nA sign the model cannot read consistently leaves the old limit in place:");
        println!("missing a sign is safer than switching to a wrong one.");
    }
    Ok(())
}
