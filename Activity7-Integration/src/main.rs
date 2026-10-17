//! Activity 7: everything together on a dashcam clip.
//!
//! Per frame:
//! 1. lane detection on the clean frame, plus YOLOX object detection;
//! 2. a roadside speed sign and a traffic light are painted into the scene on a
//!    schedule (the sample clips contain no German signs and no signals);
//! 3. the *real* classifiers read the sign and the signal from the frame's own
//!    pixels, and the trackers confirm them over several frames;
//! 4. the instrument cluster shows speed, limit, lane assist and the camera's
//!    reading, while the camera view carries the boxes, the lanes and the
//!    confirmed signal.
//!
//! ```text
//! cargo run --release -p adas_drive                         # data/video.mp4 → output/drive.mp4
//! cargo run --release -p adas_drive -- --output drive.gif --signs "2:30,9:80" --lights "5:red,12:green"
//! ```

use std::path::PathBuf;

use adas_core::cluster::{draw_cluster, ClusterState, LaneStatus};
use adas_core::detector::{DetectorConfig, ObjectDetector};
use adas_core::lanes::LaneDetector;
use adas_core::overlay;
use adas_core::render::{self, rgba, Align, Fonts, CROP_PER_RADIUS, TRAFFIC_LIGHT_HALF_HEIGHT};
use adas_core::speed::{LimitEvent, SpeedLimitTracker, TrackerConfig};
use adas_core::traffic_light::{self, LightColor, LightReading, LightTracker};
use adas_core::video::{FrameSink, FrameSource};
use adas_core::{coco, paths, Prediction, SignClassifier};
use anyhow::{bail, Context, Result};
use clap::Parser;
use image::RgbImage;
use tiny_skia::Pixmap;

#[derive(Parser)]
#[command(about = "Lanes + YOLOX detection + traffic-sign and traffic-light recognition + cluster")]
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
    /// Roadside traffic lights as "seconds:colour" pairs (red, amber, green).
    #[arg(long, default_value = "4:green,11:red,17:amber")]
    lights: String,
    /// Only render this many seconds.
    #[arg(long)]
    duration: Option<f32>,
    #[arg(long)]
    model: Option<PathBuf>,
    /// YOLOX detector for the object boxes (default: the bundled Nano model).
    #[arg(long)]
    detector: Option<PathBuf>,
    /// Skip object detection entirely.
    #[arg(long)]
    no_detection: bool,
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

/// A traffic light the car drives past.
#[derive(Copy, Clone, Debug, PartialEq)]
struct Light {
    at: f32,
    colour: LightColor,
}

/// How long a sign or light is on screen, from the horizon to leaving the frame.
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

fn parse_lights(spec: &str) -> Result<Vec<Light>> {
    let mut lights: Vec<Light> = spec
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|pair| {
            let (t, c) = pair
                .split_once(':')
                .with_context(|| format!("bad light {pair:?}, expected seconds:colour"))?;
            let colour = match c.trim().to_ascii_lowercase().as_str() {
                "red" => LightColor::Red,
                "amber" | "yellow" => LightColor::Amber,
                "green" => LightColor::Green,
                other => bail!("unknown light colour {other:?} (use red, amber or green)"),
            };
            Ok(Light { at: t.trim().parse()?, colour })
        })
        .collect::<Result<_>>()?;
    lights.sort_by(|a, b| a.at.total_cmp(&b.at));
    if lights.windows(2).any(|w| w[1].at - w[0].at < PASS_SECONDS) {
        bail!("lights must be at least {PASS_SECONDS}s apart");
    }
    Ok(lights)
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

/// Where a passing traffic light is: on the left verge, swinging out towards
/// the camera as the car goes under it.
fn light_pose(progress: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
    let p = progress.clamp(0.0, 1.0).powf(2.4);
    let x = w * (0.17 + 0.34 * p);
    let ground = h * (0.60 + 0.34 * p);
    let r = h * (0.010 + 0.070 * p);
    let y = ground - r * (3.4 + 5.6 * p);
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
    let lights = parse_lights(&args.lights)?;
    let input = args.input.clone().unwrap_or_else(paths::default_video);
    let source = FrameSource::open(&input, None, args.duration, 25.0)?;
    let info = source.info();
    let model = SignClassifier::load(args.model.clone().unwrap_or_else(paths::default_model))?;
    let fonts = Fonts::load();

    let detector = if args.no_detection {
        None
    } else {
        let path = args.detector.clone().unwrap_or_else(paths::default_detector);
        let config = DetectorConfig {
            confidence: 0.35,
            keep: Some(coco::ADAS_CLASSES.to_vec()),
            ..Default::default()
        };
        match ObjectDetector::load_with(&path, config) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("Object detection off: {e}");
                None
            }
        }
    };

    let w = args.width;
    let h = (info.height as f32 * w as f32 / info.width as f32).round() as u32 & !1;
    let cluster_h = (w as f32 * 0.24).round() as u32 & !1;
    let mut sink = FrameSink::create(&args.output, w, h + cluster_h, info.fps)?;
    let mut lanes = LaneDetector::default();
    let mut tracker = SpeedLimitTracker::new(TrackerConfig::default());
    let mut light_tracker = LightTracker::new(3);
    let mut changed_at: Option<f32> = None;
    let mut last_reading: Option<(Prediction, f32)> = None;
    let (mut n, mut objects, mut events) = (0usize, 0usize, Vec::new());
    let mut light_events: Vec<(f32, LightColor)> = Vec::new();

    for (i, frame) in source.enumerate() {
        let t = i as f32 / info.fps;
        let frame = image::imageops::resize(&frame?, w, h, image::imageops::FilterType::Triangle);

        // 1. Real detectors on the clean frame, before anything is painted in.
        let lane_frame = lanes.detect(&frame);
        let detections = match &detector {
            Some(detector) => detector.detect(&frame)?,
            None => Vec::new(),
        };
        objects += detections.len();

        // 2. Paint the passing sign and traffic light (if any) into the scene.
        let mut pix = render::pixmap_from_rgb(&frame);
        let mut sign_box = None;
        if let Some(sign) = signs.iter().find(|s| (s.at..s.at + PASS_SECONDS).contains(&t)) {
            let (sx, sy, r, ground) = sign_pose((t - sign.at) / PASS_SECONDS, w as f32, h as f32);
            render::fill_rect(&mut pix, sx - r * 0.07, sy, r * 0.14, ground - sy, rgba(150, 152, 156, 1.0));
            if sign.kmh == 0 {
                render::end_of_limit_sign(&mut pix, &fonts, sx, sy, r, 80);
            } else {
                render::speed_sign(&mut pix, &fonts, sx, sy, r, sign.kmh);
            }
            sign_box = Some((sx, sy, r));
        }
        let mut light_box = None;
        if let Some(light) = lights.iter().find(|l| (l.at..l.at + PASS_SECONDS).contains(&t)) {
            let (lx, ly, r, ground) = light_pose((t - light.at) / PASS_SECONDS, w as f32, h as f32);
            let bottom = ly + r * TRAFFIC_LIGHT_HALF_HEIGHT;
            render::fill_rect(
                &mut pix,
                lx - r * 0.16,
                bottom,
                r * 0.32,
                (ground - bottom).max(1.0),
                rgba(120, 122, 126, 1.0),
            );
            render::traffic_light_head(&mut pix, lx, ly, r, light.colour);
            light_box = Some((lx, ly, r));
        }
        let frame = render::rgb_from_pixmap(&pix);

        // 3. The camera reads the painted sign from the frame's own pixels.
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

        // 4. And the signal's colour from the head's own pixels.
        let reading = match light_box {
            Some((lx, ly, r)) if r >= 4.0 => crop_square(&frame, lx, ly, r * 6.0)
                .map(|crop| traffic_light::classify(&crop))
                .unwrap_or(LightReading { colour: LightColor::Unknown, confidence: 0.0 }),
            _ => LightReading { colour: LightColor::Unknown, confidence: 0.0 },
        };
        if light_tracker.observe(reading) {
            light_events.push((t, light_tracker.current()));
        }

        // 5. Overlays on the camera image.
        overlay::draw_lanes(&mut pix, &lane_frame);
        overlay::draw_detections(&mut pix, &fonts, &detections);
        if let (Some((sx, sy, r)), Some(p)) = (sign_box, &prediction) {
            let half = r * 1.35;
            let box_colour = rgba(255, 255, 255, 0.85);
            render::stroke_rect(&mut pix, sx - half, sy - half, half * 2.0, half * 2.0, 2.0, box_colour);
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
        overlay::light_panel(&mut pix, &fonts, light_tracker.current());
        overlay::caption(
            &mut pix,
            &fonts,
            "Real lanes + YOLOX detection · simulated signs & signals, real classifiers",
        );

        // 6. The cluster below the camera view.
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

    let fps = n as f32 / (n as f32 / info.fps).max(1e-3);
    println!("{n} frames ({fps:.0} fps of source) → {}", args.output.display());
    if detector.is_some() {
        println!("  YOLOX detection: {objects} object(s) across the clip");
    }
    for (t, event) in events {
        match event {
            LimitEvent::Set(k) => println!("  {t:5.1}s  limit {k} km/h"),
            LimitEvent::Cleared => println!("  {t:5.1}s  limit lifted"),
        }
    }
    for (t, colour) in light_events {
        println!("  {t:5.1}s  signal {colour:?}");
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
    fn lights_parse_colours_and_spacing() {
        let l = parse_lights("12:green, 5:red, 18:amber").unwrap();
        assert_eq!(
            l,
            vec![
                Light { at: 5.0, colour: LightColor::Red },
                Light { at: 12.0, colour: LightColor::Green },
                Light { at: 18.0, colour: LightColor::Amber },
            ]
        );
        assert!(parse_lights("5:purple").is_err());
        assert!(parse_lights("5:red,6:green").is_err());
        assert!(parse_lights("").unwrap().is_empty());
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
