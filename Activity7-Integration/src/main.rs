//! Activity 7: everything together on a dashcam clip.
//!
//! Per frame: lane detection → a roadside speed sign is painted into the scene
//! (the sample clips have no German signs) → the *real* classifier reads the
//! sign from the frame's pixels → the speed-limit tracker confirms it → the
//! instrument cluster shows speed, limit, lane assist and the camera's reading.
//!
//! ```text
//! cargo run --release -p adas_drive                         # data/video.mp4 → output/drive.mp4
//! cargo run --release -p adas_drive -- --output drive.gif --signs "2:30,9:80,16:70,22:120"
//! ```

use std::path::PathBuf;

use adas_core::cluster::{draw_cluster, ClusterState, LaneStatus};
use adas_core::lanes::LaneDetector;
use adas_core::overlay;
use adas_core::render::{self, rgba, Align, Fonts, CROP_PER_RADIUS};
use adas_core::speed::{LimitEvent, SpeedLimitTracker, TrackerConfig};
use adas_core::video::{FrameSink, FrameSource};
use adas_core::{paths, Prediction, SignClassifier};
use anyhow::{bail, Context, Result};
use clap::Parser;
use image::RgbImage;
use tiny_skia::Pixmap;

#[derive(Parser)]
#[command(about = "Lanes + traffic-sign recognition + instrument cluster on a dashcam video")]
struct Args {
    /// Dashcam video or folder of frames (default: data/video.mp4).
    #[arg(long)]
    input: Option<PathBuf>,
    /// .mp4, .webm, .gif or a folder of PNGs.
    #[arg(long, default_value = "output/drive.mp4")]
    output: PathBuf,
    /// Roadside signs as "seconds:km/h" pairs. 0 km/h = "end of 80".
    #[arg(long, default_value = "1.5:80,8:120,14.5:70,21:30")]
    signs: String,
    /// Only render this many seconds.
    #[arg(long)]
    duration: Option<f32>,
    #[arg(long)]
    model: Option<PathBuf>,
    /// Output width (the video is scaled; the cluster is added below it).
    #[arg(long, default_value_t = 960)]
    width: u32,
}

/// A sign the car drives past, appearing at `at` seconds.
#[derive(Copy, Clone, Debug, PartialEq)]
struct Sign {
    at: f32,
    kmh: u32,
}

/// How long a sign is on screen, from the horizon to leaving the frame.
const PASS_SECONDS: f32 = 2.8;

fn parse_signs(spec: &str) -> Result<Vec<Sign>> {
    let mut signs: Vec<Sign> = spec
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|pair| {
            let (t, k) =
                pair.split_once(':').with_context(|| format!("bad sign {pair:?}, expected seconds:kmh"))?;
            Ok(Sign { at: t.trim().parse()?, kmh: k.trim().parse()? })
        })
        .collect::<Result<_>>()?;
    signs.sort_by(|a, b| a.at.total_cmp(&b.at));
    if signs.windows(2).any(|w| w[1].at - w[0].at < PASS_SECONDS) {
        bail!("signs must be at least {PASS_SECONDS}s apart");
    }
    Ok(signs)
}

/// Where a passing sign is on screen: it starts small near the vanishing
/// point and sweeps out to the right edge as the car approaches.
/// Returns (centre x, centre y, radius, ground y) in frame pixels.
fn sign_pose(progress: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
    let p = progress.clamp(0.0, 1.0).powf(2.2); // perspective: slow far away, fast up close
    let x = w * (0.585 + 0.43 * p);
    let ground = h * (0.62 + 0.30 * p);
    let r = h * (0.008 + 0.075 * p);
    let y = ground - r * (2.2 + 5.5 * p); // on a post
    (x, y, r, ground)
}

/// The driver's speed: they start a little fast and adjust to each new limit
/// about a second and a half after the sign, so the cluster briefly shows the
/// over-speed warning after a limit drops.
fn ego_speed(t: f32, signs: &[Sign]) -> f32 {
    const START: f32 = 86.0;
    const REACTION: f32 = 1.5; // seconds after the sign appears
    const TAU: f32 = 1.1; // how quickly speed settles (s)
    let (mut speed, mut target, mut since) = (START, START, 0.0);
    for sign in signs.iter().filter(|s| s.at + REACTION <= t) {
        let react = sign.at + REACTION;
        speed += (target - speed) * (1.0 - (-(react - since) / TAU).exp());
        target = if sign.kmh == 0 { 105.0 } else { sign.kmh as f32 + 2.0 };
        since = react;
    }
    speed += (target - speed) * (1.0 - (-(t - since) / TAU).exp());
    speed + 1.2 * (t * 0.9).sin()
}

fn crop_square(frame: &RgbImage, cx: f32, cy: f32, side: f32) -> Option<RgbImage> {
    let half = side / 2.0;
    let (x0, y0) = ((cx - half).round(), (cy - half).round());
    if x0 < 0.0 || y0 < 0.0 || x0 + side > frame.width() as f32 || y0 + side > frame.height() as f32 {
        return None; // partly off-screen: no clean view of the sign
    }
    let s = side.round() as u32;
    Some(image::imageops::crop_imm(frame, x0 as u32, y0 as u32, s, s).to_image())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let signs = parse_signs(&args.signs)?;
    let input = args.input.clone().unwrap_or_else(paths::default_video);
    let source = FrameSource::open(&input, None, args.duration, 25.0)?;
    let info = source.info();
    let model = SignClassifier::load(args.model.clone().unwrap_or_else(paths::default_model))?;
    let fonts = Fonts::load();

    let w = args.width;
    let h = (info.height as f32 * w as f32 / info.width as f32).round() as u32 & !1;
    let cluster_h = (w as f32 * 0.24).round() as u32 & !1;
    let mut sink = FrameSink::create(&args.output, w, h + cluster_h, info.fps)?;
    let mut lanes = LaneDetector::default();
    let mut tracker = SpeedLimitTracker::new(TrackerConfig::default());
    let mut changed_at: Option<f32> = None;
    let mut last_reading: Option<(Prediction, f32)> = None;
    let (mut n, mut events) = (0usize, Vec::new());

    for (i, frame) in source.enumerate() {
        let t = i as f32 / info.fps;
        let mut frame = image::imageops::resize(&frame?, w, h, image::imageops::FilterType::Triangle);
        let lane_frame = lanes.detect(&frame); // before painting signs into the scene

        // 1. Paint the passing sign (if any) into the scene.
        let mut pix = render::pixmap_from_rgb(&frame);
        let passing = signs.iter().find(|s| (s.at..s.at + PASS_SECONDS).contains(&t));
        let mut sign_box = None;
        if let Some(sign) = passing {
            let (sx, sy, r, ground) = sign_pose((t - sign.at) / PASS_SECONDS, w as f32, h as f32);
            render::fill_rect(&mut pix, sx - r * 0.07, sy, r * 0.14, ground - sy, rgba(150, 152, 156, 1.0));
            if sign.kmh == 0 {
                render::end_of_limit_sign(&mut pix, &fonts, sx, sy, r, 80);
            } else {
                render::speed_sign(&mut pix, &fonts, sx, sy, r, sign.kmh);
            }
            sign_box = Some((sx, sy, r));
        }
        frame = render::rgb_from_pixmap(&pix);

        // 2. The camera reads the sign from the frame's own pixels.
        let prediction = match sign_box {
            Some((sx, sy, r)) if r >= 6.0 => crop_square(&frame, sx, sy, r * CROP_PER_RADIUS)
                .map(|crop| model.classify(&crop))
                .transpose()?,
            _ => None,
        };
        if let Some(event) = tracker.observe(prediction.as_ref()) {
            changed_at = Some(t);
            events.push((t, event));
        }
        if let Some(p) = &prediction {
            last_reading = Some((p.clone(), t));
        }

        // 3. Overlays on the camera image.
        overlay::draw_lanes(&mut pix, &lane_frame);
        if let (Some((sx, sy, r)), Some(p)) = (sign_box, &prediction) {
            let half = r * 1.35;
            let box_colour = rgba(255, 255, 255, 0.85);
            render::polyline(
                &mut pix,
                &[
                    (sx - half, sy - half),
                    (sx + half, sy - half),
                    (sx + half, sy + half),
                    (sx - half, sy + half),
                    (sx - half, sy - half),
                ],
                2.0,
                box_colour,
                false,
            );
            let tag = format!(
                "{} · {:.0}%",
                p.label().replace("Speed limit ", "").replace(['(', ')'], ""),
                p.confidence * 100.0
            );
            let size = (h as f32 * 0.03).max(13.0);
            render::text(
                &mut pix,
                &fonts.semibold,
                &tag,
                sx + half,
                sy - half - size * 0.4,
                size,
                box_colour,
                Align::Right,
            );
        }
        overlay::caption(&mut pix, &fonts, "Simulated roadside signs · real classifier");

        // 4. The cluster below the camera view.
        let camera = last_reading
            .as_ref()
            .filter(|(_, seen)| t - seen < 0.6)
            .map(|(p, _)| (p.label().replace("Speed limit ", "").replace(['(', ')'], ""), p.confidence));
        let state = ClusterState {
            speed_kmh: ego_speed(t, &signs),
            limit: tracker.current(),
            limit_changed_at: changed_at,
            lanes: LaneStatus {
                left: lane_frame.left.is_some(),
                right: lane_frame.right.is_some(),
                departure: lane_frame.departure,
            },
            camera,
            time_s: t,
        };
        let mut canvas = Pixmap::new(w, h + cluster_h).context("canvas")?;
        canvas.fill(rgba(6, 7, 9, 1.0));
        canvas.draw_pixmap(
            0,
            0,
            pix.as_ref(),
            &tiny_skia::PixmapPaint::default(),
            tiny_skia::Transform::identity(),
            None,
        );
        let pad = cluster_h as f32 * 0.06;
        draw_cluster(
            &mut canvas,
            &fonts,
            pad,
            h as f32 + pad,
            w as f32 - 2.0 * pad,
            cluster_h as f32 - 2.0 * pad,
            &state,
        );
        sink.write(&render::rgb_from_pixmap(&canvas))?;
        n += 1;
    }
    sink.finish()?;

    println!("{n} frames → {}", args.output.display());
    for (t, event) in events {
        match event {
            LimitEvent::Set(k) => println!("  {t:5.1}s  limit {k} km/h"),
            LimitEvent::Cleared => println!("  {t:5.1}s  limit lifted"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signs_parse_sorted_and_spaced() {
        let s = parse_signs("9:80, 2:30").unwrap();
        assert_eq!(s, vec![Sign { at: 2.0, kmh: 30 }, Sign { at: 9.0, kmh: 80 }]);
        assert!(parse_signs("1:30,2:50").is_err());
        assert!(parse_signs("oops").is_err());
    }

    #[test]
    fn a_sign_grows_and_moves_right_as_we_approach() {
        let (x0, _, r0, _) = sign_pose(0.0, 960.0, 540.0);
        let (x1, _, r1, _) = sign_pose(0.7, 960.0, 540.0);
        assert!(x1 > x0 && r1 > r0 * 3.0);
    }

    #[test]
    fn the_driver_reacts_to_new_limits_late() {
        let signs = [Sign { at: 2.0, kmh: 50 }];
        assert!(ego_speed(3.0, &signs) > 75.0); // not yet reacted: over the new limit
        assert!((ego_speed(9.0, &signs) - 52.0).abs() < 3.0); // settled near the limit
    }
}
