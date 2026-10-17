//! Lane detection in pure Rust (no OpenCV): paint mask → Canny → region of
//! interest → banded Hough transform → one line per side → smoothing.
//!
//! Compared with the original Activity 1 this (a) fits *one* left and *one*
//! right lane line instead of drawing every Hough segment, (b) only accepts
//! lane-like angles, so horizon edges and car bonnets are ignored, (c) smooths
//! over time and survives dashed-line gaps, and (d) reports lane departure.

use image::{imageops::FilterType, GrayImage, Luma, RgbImage};

/// Tunable parameters. The defaults suit a centred dashcam, 16:9.
#[derive(Clone, Debug)]
pub struct LaneConfig {
    /// Region of interest as fractions of the frame: bottom-left, top-left,
    /// top-right, bottom-right. (The original ROI was lopsided: 3/7 vs 9/12.)
    pub roi: [(f32, f32); 4],
    /// Detection runs on a copy scaled to this width (speed; ~10x fewer pixels).
    pub work_width: u32,
    /// A pixel is "white paint" if its darkest channel is at least this bright.
    pub white_threshold: u8,
    pub canny_low: f32,
    pub canny_high: f32,
    /// Normal angle bands (degrees) for the left and right lane lines.
    pub left_band: (u32, u32),
    pub right_band: (u32, u32),
    /// Minimum Hough votes, as a fraction of the ROI height in work pixels.
    pub min_votes_frac: f32,
    /// Exponential smoothing factor for new measurements (0..1, higher = snappier).
    pub smoothing: f32,
    /// Keep the last good line this many frames through dashed gaps.
    pub hold_frames: u32,
    /// Where the camera sits horizontally (0.5 = centred).
    pub camera_x: f32,
    /// Offset (fraction of lane width) beyond which we warn about departure.
    pub departure_threshold: f32,
}

impl Default for LaneConfig {
    fn default() -> Self {
        Self {
            roi: [(0.06, 0.97), (0.44, 0.61), (0.56, 0.61), (0.96, 0.97)],
            work_width: 480,
            white_threshold: 175,
            canny_low: 50.0,
            canny_high: 150.0,
            // Wide enough for the near line when the car drifts onto it;
            // near-horizontal edges (normal ≈ 90°) are still excluded.
            left_band: (14, 76),
            right_band: (104, 166),
            min_votes_frac: 0.12,
            smoothing: 0.35,
            hold_frames: 8,
            camera_x: 0.5,
            departure_threshold: 0.22,
        }
    }
}

/// A lane line from the bottom of the region of interest to its top, in frame pixels.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Lane {
    pub bottom: (f32, f32),
    pub top: (f32, f32),
}

impl Lane {
    fn lerp(self, other: Lane, t: f32) -> Lane {
        let mix = |a: f32, b: f32| a + (b - a) * t;
        Lane {
            bottom: (mix(self.bottom.0, other.bottom.0), self.bottom.1),
            top: (mix(self.top.0, other.top.0), self.top.1),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Departure {
    None,
    /// Drifting towards (or over) the left line.
    Left,
    /// Drifting towards (or over) the right line.
    Right,
}

/// What the detector reports for one frame.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LaneFrame {
    pub left: Option<Lane>,
    pub right: Option<Lane>,
    /// Car offset from the lane centre as a fraction of lane width (+ = right).
    pub offset: Option<f32>,
    pub departure: Departure,
}

/// Stateful detector (keeps smoothed lines between frames).
pub struct LaneDetector {
    pub config: LaneConfig,
    left: Option<(Lane, u32)>,
    right: Option<(Lane, u32)>,
    /// Smoothed lane width at the bottom of the ROI, for one-line frames.
    lane_width: Option<f32>,
}

impl LaneDetector {
    pub fn new(config: LaneConfig) -> Self {
        Self { config, left: None, right: None, lane_width: None }
    }

    /// Forget previous frames (e.g. when a new video starts).
    pub fn reset(&mut self) {
        self.left = None;
        self.right = None;
        self.lane_width = None;
    }

    pub fn detect(&mut self, frame: &RgbImage) -> LaneFrame {
        let (w, h) = frame.dimensions();
        let scale = (w as f32 / self.config.work_width as f32).max(1.0);
        let (ww, wh) = ((w as f32 / scale).round() as u32, (h as f32 / scale).round() as u32);
        let small = image::imageops::resize(frame, ww, wh, FilterType::Triangle);
        let edges = self.roi_edges(&small);

        let roi_top = self.config.roi[1].1.min(self.config.roi[2].1) * wh as f32;
        let roi_bottom = self.config.roi[0].1.max(self.config.roi[3].1) * wh as f32;
        let min_votes = ((roi_bottom - roi_top) * self.config.min_votes_frac).max(8.0) as u32;
        let to_full = |lane: Lane| Lane {
            bottom: (lane.bottom.0 * scale, lane.bottom.1 * scale),
            top: (lane.top.0 * scale, lane.top.1 * scale),
        };

        let mid = ww as f32 * self.config.camera_x;
        let left = fit_lane(&edges, self.config.left_band, min_votes, roi_top, roi_bottom)
            .filter(|l| l.bottom.0 < mid + 0.1 * ww as f32 && l.top.0 < mid + 0.15 * ww as f32)
            .map(to_full);
        let right = fit_lane(&edges, self.config.right_band, min_votes, roi_top, roi_bottom)
            .filter(|l| l.bottom.0 > mid - 0.1 * ww as f32 && l.top.0 > mid - 0.15 * ww as f32)
            .map(to_full);

        let (alpha, hold) = (self.config.smoothing, self.config.hold_frames);
        let left = smooth(&mut self.left, left, alpha, hold);
        let right = smooth(&mut self.right, right, alpha, hold);

        // Lane centre from both lines, or from one line plus the remembered
        // lane width (when drifting, the far line often leaves the ROI).
        let (centre, width) = match (left, right, self.lane_width) {
            (Some(l), Some(r), _) if r.bottom.0 - l.bottom.0 > 0.15 * w as f32 => {
                let width = r.bottom.0 - l.bottom.0;
                self.lane_width = Some(match self.lane_width {
                    Some(prev) => prev + (width - prev) * 0.1,
                    None => width,
                });
                (Some((l.bottom.0 + r.bottom.0) / 2.0), Some(width))
            }
            (Some(l), None, Some(width)) => (Some(l.bottom.0 + width / 2.0), Some(width)),
            (None, Some(r), Some(width)) => (Some(r.bottom.0 - width / 2.0), Some(width)),
            _ => (None, None),
        };
        let offset = centre.zip(width).map(|(c, wd)| (w as f32 * self.config.camera_x - c) / wd);
        let departure = match offset {
            Some(o) if o < -self.config.departure_threshold => Departure::Left,
            Some(o) if o > self.config.departure_threshold => Departure::Right,
            _ => Departure::None,
        };
        LaneFrame { left, right, offset, departure }
    }

    /// Lane-paint edges inside the region of interest (at work resolution).
    /// Public so the lane demo can show what the detector "sees".
    pub fn roi_edges(&self, small: &RgbImage) -> GrayImage {
        let paint = paint_mask(small, self.config.white_threshold);
        let mut edges = imageproc::edges::canny(&paint, self.config.canny_low, self.config.canny_high);
        let (w, h) = edges.dimensions();
        let roi: Vec<(f32, f32)> =
            self.config.roi.iter().map(|&(x, y)| (x * w as f32, y * h as f32)).collect();
        for (x, y, px) in edges.enumerate_pixels_mut() {
            if px[0] > 0 && !inside_convex(&roi, x as f32 + 0.5, y as f32 + 0.5) {
                *px = Luma([0]);
            }
        }
        edges
    }
}

impl Default for LaneDetector {
    fn default() -> Self {
        Self::new(LaneConfig::default())
    }
}

/// White or yellow road paint → 255, everything else → 0.
pub fn paint_mask(img: &RgbImage, white_threshold: u8) -> GrayImage {
    GrayImage::from_fn(img.width(), img.height(), |x, y| {
        let [r, g, b] = img.get_pixel(x, y).0;
        let (lo, hi) = (r.min(g).min(b), r.max(g).max(b));
        let white = lo >= white_threshold && hi - lo < 70;
        let yellow = r >= 140 && g >= 110 && (b as i16) < g as i16 - 50 && r as i16 - b as i16 > 70;
        Luma([if white || yellow { 255 } else { 0 }])
    })
}

/// Point-in-convex-polygon (vertices in order).
fn inside_convex(poly: &[(f32, f32)], x: f32, y: f32) -> bool {
    let mut sign = 0.0f32;
    for i in 0..poly.len() {
        let (x1, y1) = poly[i];
        let (x2, y2) = poly[(i + 1) % poly.len()];
        let cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1);
        if cross != 0.0 {
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
    }
    true
}

/// Hough transform restricted to one band of normal angles; returns the
/// vote-weighted average of the strongest lines, as a segment between the
/// ROI's bottom and top rows.
fn fit_lane(edges: &GrayImage, band: (u32, u32), min_votes: u32, top: f32, bottom: f32) -> Option<Lane> {
    let (w, h) = edges.dimensions();
    let rmax = ((w * w + h * h) as f32).sqrt().ceil() as i32;
    let angles: Vec<(u32, f32, f32)> = (band.0..=band.1)
        .map(|deg| {
            let (s, c) = (deg as f32).to_radians().sin_cos();
            (deg, s, c)
        })
        .collect();
    let bins = (2 * rmax + 1) as usize;
    let mut acc = vec![0u32; angles.len() * bins];
    for (x, y, px) in edges.enumerate_pixels() {
        if px[0] == 0 {
            continue;
        }
        for (i, &(_, s, c)) in angles.iter().enumerate() {
            let r = (x as f32 * c + y as f32 * s).round() as i32 + rmax;
            acc[i * bins + r as usize] += 1;
        }
    }

    // Strongest bins with simple non-maximum suppression.
    let mut peaks: Vec<(u32, usize, usize)> = acc
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v >= min_votes)
        .map(|(k, &v)| (v, k / bins, k % bins))
        .collect();
    peaks.sort_unstable_by_key(|p| std::cmp::Reverse(p.0));
    let mut kept: Vec<(u32, usize, usize)> = Vec::new();
    for p in peaks {
        if kept.len() >= 6 {
            break;
        }
        let near = kept.iter().any(|k| k.1.abs_diff(p.1) <= 3 && k.2.abs_diff(p.2) <= 8);
        if !near {
            kept.push(p);
        }
    }
    let best = kept.first()?.0 as f32;
    let (mut sum_w, mut bx, mut tx) = (0.0f32, 0.0f32, 0.0f32);
    for &(votes, ai, ri) in kept.iter().filter(|k| k.0 as f32 >= 0.6 * best) {
        let (_, s, c) = angles[ai];
        let r = ri as f32 - rmax as f32;
        let x_at = |y: f32| (r - y * s) / c;
        let weight = votes as f32;
        bx += weight * x_at(bottom);
        tx += weight * x_at(top);
        sum_w += weight;
    }
    let lane = Lane { bottom: (bx / sum_w, bottom), top: (tx / sum_w, top) };
    let span = w as f32;
    (lane.bottom.0.is_finite() && lane.bottom.0 > -span && lane.bottom.0 < 2.0 * span).then_some(lane)
}

fn smooth(state: &mut Option<(Lane, u32)>, measured: Option<Lane>, alpha: f32, hold: u32) -> Option<Lane> {
    *state = match (*state, measured) {
        (Some((prev, _)), Some(new)) => Some((prev.lerp(new, alpha), 0)),
        (None, Some(new)) => Some((new, 0)),
        (Some((prev, missed)), None) if missed < hold => Some((prev, missed + 1)),
        _ => None,
    };
    state.map(|(lane, _)| lane)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    use imageproc::drawing::draw_polygon_mut;
    use imageproc::point::Point;

    /// A dark road with two painted lines converging towards the horizon.
    fn road(shift: i32) -> RgbImage {
        let mut img = RgbImage::from_pixel(960, 540, Rgb([70, 72, 75]));
        let paint = |img: &mut RgbImage, bx: i32, tx: i32, colour: Rgb<u8>| {
            let pts = [
                Point::new(bx - 9 + shift, 530),
                Point::new(tx - 3 + shift, 340),
                Point::new(tx + 3 + shift, 340),
                Point::new(bx + 9 + shift, 530),
            ];
            draw_polygon_mut(img, &pts, colour);
        };
        paint(&mut img, 190, 440, Rgb([230, 200, 40])); // yellow left line
        paint(&mut img, 770, 520, Rgb([240, 240, 240])); // white right line
        img
    }

    fn x_at(lane: Lane, y: f32) -> f32 {
        let t = (y - lane.bottom.1) / (lane.top.1 - lane.bottom.1);
        lane.bottom.0 + (lane.top.0 - lane.bottom.0) * t
    }

    #[test]
    fn finds_both_lines_on_a_synthetic_road() {
        let mut detector = LaneDetector::default();
        let frame = detector.detect(&road(0));
        let (left, right) = (frame.left.expect("left lane"), frame.right.expect("right lane"));
        // At y=500 the painted left line is at ~229 and the right at ~731.
        assert!((x_at(left, 500.0) - 229.0).abs() < 20.0, "left at {}", x_at(left, 500.0));
        assert!((x_at(right, 500.0) - 731.0).abs() < 20.0, "right at {}", x_at(right, 500.0));
        assert_eq!(frame.departure, Departure::None);
        assert!(frame.offset.unwrap().abs() < 0.1);
    }

    #[test]
    fn reports_departure_when_the_car_drifts() {
        // Lines slide left on screen = the car is drifting right.
        let mut detector = LaneDetector::default();
        detector.detect(&road(0)); // learns the lane width
        let mut last = None;
        for shift in (0..=170).step_by(10) {
            last = Some(detector.detect(&road(-shift)));
        }
        let frame = last.unwrap();
        assert_eq!(frame.departure, Departure::Right, "offset {:?}", frame.offset);

        let mut detector = LaneDetector::default();
        detector.detect(&road(0));
        for shift in (0..=170).step_by(10) {
            last = Some(detector.detect(&road(shift)));
        }
        assert_eq!(last.unwrap().departure, Departure::Left);
    }

    #[test]
    fn holds_lines_through_short_gaps() {
        let mut detector = LaneDetector::default();
        detector.detect(&road(0));
        let blank = RgbImage::from_pixel(960, 540, Rgb([70, 72, 75]));
        for _ in 0..detector.config.hold_frames {
            assert!(detector.detect(&blank).left.is_some());
        }
        assert!(detector.detect(&blank).left.is_none());
    }

    #[test]
    fn convex_test() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(inside_convex(&square, 5.0, 5.0));
        assert!(!inside_convex(&square, 15.0, 5.0));
    }
}
