//! Activity 6: the instrument cluster.
//!
//! Renders the cluster from `adas_core::cluster` through a short scripted
//! scenario, or a single snapshot of any state you choose.
//!
//! ```text
//! cargo run --release -p instrument_cluster                       # output/cluster.gif
//! cargo run --release -p instrument_cluster -- --snapshot over.png --speed 66 --limit 50
//! ```

use std::path::PathBuf;

use adas_core::cluster::{draw_cluster, ClusterState, LaneStatus};
use adas_core::lanes::Departure;
use adas_core::render::{self, rgba, Fonts};
use adas_core::video::FrameSink;
use anyhow::Result;
use clap::{Parser, ValueEnum};

#[derive(Parser)]
#[command(about = "Render the ADAS instrument cluster")]
struct Args {
    /// Animated scenario output: .gif, .mp4, .webm or a folder of PNGs.
    #[arg(long, default_value = "output/cluster.gif")]
    output: PathBuf,
    /// Render one still to this PNG instead of the scenario.
    #[arg(long)]
    snapshot: Option<PathBuf>,
    #[arg(long, default_value_t = 960)]
    width: u32,
    /// Snapshot: current speed in km/h.
    #[arg(long, default_value_t = 48.0)]
    speed: f32,
    /// Snapshot: speed limit in force.
    #[arg(long)]
    limit: Option<u32>,
    /// Snapshot: lane departure.
    #[arg(long, value_enum, default_value_t = Drift::None)]
    departure: Drift,
}

#[derive(Copy, Clone, ValueEnum)]
enum Drift {
    None,
    Left,
    Right,
}

/// Interpolation through (time, value) keyframes, eased in and out.
fn keyframes(points: &[(f32, f32)], t: f32) -> f32 {
    let i = points.iter().position(|&(k, _)| k > t).unwrap_or(points.len());
    match i {
        0 => points[0].1,
        i if i == points.len() => points[i - 1].1,
        i => {
            let ((t0, v0), (t1, v1)) = (points[i - 1], points[i]);
            let u = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            let u = u * u * (3.0 - 2.0 * u);
            v0 + (v1 - v0) * u
        }
    }
}

/// A 14 s story: pulling away, a 50 sign, creeping over, a drift, back to normal.
fn scenario(t: f32) -> ClusterState {
    let speed = keyframes(
        &[(0.0, 0.0), (2.5, 46.0), (5.0, 49.0), (7.0, 58.0), (8.5, 66.0), (10.0, 50.0), (14.0, 48.0)],
        t,
    );
    let (limit, changed) = if t >= 2.0 { (Some(50), Some(2.0)) } else { (None, None) };
    let departure = if (11.0..12.6).contains(&t) { Departure::Left } else { Departure::None };
    let camera = (1.2..3.2).contains(&t).then(|| ("Speed limit (50km/h)".to_string(), 0.99));
    ClusterState {
        speed_kmh: speed,
        limit,
        limit_changed_at: changed,
        lanes: LaneStatus { left: t > 0.6, right: t > 0.9, departure },
        camera,
        time_s: t,
    }
}

fn frame(fonts: &Fonts, width: u32, state: &ClusterState) -> image::RgbImage {
    let height = (width as f32 * 0.26).round() as u32;
    let mut pix = tiny_skia::Pixmap::new(width, height).expect("size");
    pix.fill(rgba(6, 7, 9, 1.0));
    let pad = height as f32 * 0.06;
    draw_cluster(&mut pix, fonts, pad, pad, width as f32 - 2.0 * pad, height as f32 - 2.0 * pad, state);
    render::rgb_from_pixmap(&pix)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let fonts = Fonts::load();

    if let Some(path) = &args.snapshot {
        let departure = match args.departure {
            Drift::None => Departure::None,
            Drift::Left => Departure::Left,
            Drift::Right => Departure::Right,
        };
        let state = ClusterState {
            speed_kmh: args.speed,
            limit: args.limit,
            lanes: LaneStatus { left: true, right: true, departure },
            time_s: 0.1,
            ..ClusterState::default()
        };
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        frame(&fonts, args.width, &state).save(path)?;
        println!("Wrote {}", path.display());
        return Ok(());
    }

    let fps = 25.0;
    let first = frame(&fonts, args.width, &scenario(0.0));
    let mut sink = FrameSink::create(&args.output, first.width(), first.height(), fps)?;
    for i in 0..(14.0 * fps) as usize {
        sink.write(&frame(&fonts, args.width, &scenario(i as f32 / fps)))?;
    }
    sink.finish()?;
    println!("Wrote {}", args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use adas_core::cluster::{speed_band, SpeedBand};

    #[test]
    fn keyframes_interpolate_and_clamp() {
        let k = [(0.0, 0.0), (2.0, 100.0)];
        assert_eq!(keyframes(&k, -1.0), 0.0);
        assert_eq!(keyframes(&k, 1.0), 50.0);
        assert_eq!(keyframes(&k, 5.0), 100.0);
    }

    #[test]
    fn the_scenario_goes_over_the_limit_and_drifts() {
        let over = scenario(8.5);
        assert_eq!(speed_band(over.speed_kmh, over.limit), SpeedBand::WayOver);
        assert_eq!(scenario(11.5).lanes.departure, Departure::Left);
        assert_eq!(speed_band(scenario(13.5).speed_kmh, Some(50)), SpeedBand::Ok);
    }
}
