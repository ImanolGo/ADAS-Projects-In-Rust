//! Traffic-light colour recognition.
//!
//! Two ways in:
//! - [`classify`] reads the colour of a crop (typically a YOLOX "traffic light"
//!   box) as RED, AMBER, GREEN or OFF.
//! - [`detect_lights`] is the classical fallback: no model, just find bright,
//!   colourful blobs and classify them, so Activity 5 runs on a bare checkout.
//!
//! Like the speed-limit tracker, a single frame is never trusted: [`LightTracker`]
//! waits for several frames to agree before it changes state.

use image::RgbImage;

use crate::detector::BoundingBox;

/// The four states plus "no idea".
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LightColor {
    Red,
    Amber,
    Green,
    /// A lamp is visible but not lit (grey lens).
    Off,
    /// Nothing light-like in the crop at all.
    Unknown,
}

impl LightColor {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Red => "RED",
            Self::Amber => "AMBER",
            Self::Green => "GREEN",
            Self::Off => "OFF",
            Self::Unknown => "—",
        }
    }

    /// RGB for drawing.
    pub fn rgb(self) -> (u8, u8, u8) {
        match self {
            Self::Red => (255, 59, 48),
            Self::Amber => (255, 176, 0),
            Self::Green => (52, 199, 89),
            Self::Off => (142, 142, 147),
            Self::Unknown => (110, 116, 124),
        }
    }
}

/// A colour reading for one crop.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LightReading {
    pub colour: LightColor,
    /// Share of the colourful weight the winning colour took, 0..=1.
    pub confidence: f32,
}

impl LightReading {
    pub fn is_lit(&self) -> bool {
        matches!(self.colour, LightColor::Red | LightColor::Amber | LightColor::Green)
    }
}

/// Hue/saturation/value, hue in degrees 0..360, the rest 0..1.
fn hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (r, g, b) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let sat = if max <= 0.0 { 0.0 } else { delta / max };
    let hue = if delta <= 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    (if hue < 0.0 { hue + 360.0 } else { hue }, sat, max)
}

/// Which of the three lamps does this hue belong to? `None` for white/grey.
fn hue_slot(hue: f32) -> Option<usize> {
    if hue < 25.0 || hue >= 335.0 {
        Some(0) // red
    } else if hue < 70.0 {
        Some(1) // amber
    } else if hue < 180.0 {
        Some(2) // green
    } else {
        None // cyan/blue/violet: not a lamp colour
    }
}

/// Minimum mean "colourfulness" before we call a crop lit at all.
const LIT_DENSITY: f32 = 0.02;

/// Classify the lamp colour of one crop.
pub fn classify(crop: &RgbImage) -> LightReading {
    let mut weights = [0.0f32; 3];
    for p in crop.pixels() {
        let (hue, sat, val) = hsv(p[0], p[1], p[2]);
        if val < 0.35 {
            continue; // too dark to be a lit lamp
        }
        if let Some(slot) = hue_slot(hue) {
            weights[slot] += sat * sat * val;
        }
    }
    let total: f32 = weights.iter().sum();
    let density = total / (crop.width() * crop.height()).max(1) as f32;
    if density < LIT_DENSITY {
        return LightReading { colour: LightColor::Off, confidence: 0.0 };
    }
    let (slot, &best) =
        weights.iter().enumerate().fold((0, &weights[0]), |acc, (i, w)| if w > acc.1 { (i, w) } else { acc });
    let colour = match slot {
        0 => LightColor::Red,
        1 => LightColor::Amber,
        _ => LightColor::Green,
    };
    LightReading { colour, confidence: best / total }
}

/// Tunables for the model-free detector.
#[derive(Clone, Debug)]
pub struct LightConfig {
    /// Minimum value (brightness) for a pixel to be a lamp.
    pub value_min: f32,
    /// Minimum saturation, so white headlights and sky are ignored.
    pub saturation_min: f32,
    /// Smallest blob to report, in pixels.
    pub min_area: usize,
    /// Widest allowed width/height ratio of a blob.
    pub max_aspect: f32,
    /// Blob must fill at least this share of its bounding box (lamps are round).
    pub min_fill: f32,
    /// Keep at most this many blobs, largest first.
    pub max_candidates: usize,
}

impl Default for LightConfig {
    fn default() -> Self {
        Self {
            value_min: 0.72,
            saturation_min: 0.55,
            min_area: 36,
            max_aspect: 2.2,
            min_fill: 0.3,
            max_candidates: 10,
        }
    }
}

/// A located, classified lamp.
#[derive(Clone, Debug, PartialEq)]
pub struct LightCandidate {
    pub bbox: BoundingBox,
    pub reading: LightReading,
}

/// Find bright colourful blobs and classify each. Model-free fallback.
pub fn detect_lights(image: &RgbImage, cfg: &LightConfig) -> Vec<LightCandidate> {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let mut candidate = vec![false; w * h];
    for (i, p) in image.pixels().enumerate() {
        let (_, sat, val) = hsv(p[0], p[1], p[2]);
        candidate[i] = val >= cfg.value_min && sat >= cfg.saturation_min;
    }

    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !candidate[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
        let mut area = 0usize;
        let mut weights = [0.0f32; 3];
        while let Some(idx) = stack.pop() {
            let (x, y) = (idx % w, idx / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            area += 1;
            let p = image.get_pixel(x as u32, y as u32);
            let (hue, sat, val) = hsv(p[0], p[1], p[2]);
            if let Some(slot) = hue_slot(hue) {
                weights[slot] += sat * sat * val;
            }
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let n = ny as usize * w + nx as usize;
                if candidate[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        let (bw, bh) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);
        let aspect = (bw / bh).max(bh / bw);
        let fill = area as f32 / (bw * bh);
        if area < cfg.min_area || aspect > cfg.max_aspect || fill < cfg.min_fill {
            continue;
        }
        let total: f32 = weights.iter().sum();
        let reading = if total <= 0.0 {
            LightReading { colour: LightColor::Off, confidence: 0.0 }
        } else {
            let (slot, &best) = weights.iter().enumerate().fold((0, &weights[0]), |acc, (i, weight)| {
                if weight > acc.1 {
                    (i, weight)
                } else {
                    acc
                }
            });
            let colour = match slot {
                0 => LightColor::Red,
                1 => LightColor::Amber,
                _ => LightColor::Green,
            };
            LightReading { colour, confidence: best / total }
        };
        out.push(LightCandidate {
            bbox: BoundingBox { x1: x0 as f32, y1: y0 as f32, x2: (x1 + 1) as f32, y2: (y1 + 1) as f32 },
            reading,
        });
    }
    out.sort_by(|a, b| b.bbox.area().total_cmp(&a.bbox.area()));
    out.truncate(cfg.max_candidates);
    out
}

/// Debounce: only change state after `needed` fresh frames agree, and drop back
/// to `Unknown` once the light has been out of view for that many frames.
#[derive(Clone, Debug)]
pub struct LightTracker {
    current: LightColor,
    streak: LightColor,
    streak_len: usize,
    lost: usize,
    needed: usize,
}

impl LightTracker {
    /// Start `Unknown`; require `needed` agreeing frames to switch.
    pub fn new(needed: usize) -> Self {
        Self {
            current: LightColor::Unknown,
            streak: LightColor::Unknown,
            streak_len: 0,
            lost: 0,
            needed: needed.max(1),
        }
    }

    pub fn current(&self) -> LightColor {
        self.current
    }

    /// Feed one frame's reading. Returns `true` if the confirmed state changed.
    pub fn observe(&mut self, reading: LightReading) -> bool {
        if !reading.is_lit() {
            self.streak_len = 0;
            self.streak = LightColor::Unknown;
            self.lost += 1;
            if self.lost >= self.needed && self.current != LightColor::Unknown {
                self.current = LightColor::Unknown;
                self.lost = 0;
                return true;
            }
            return false; // a dark frame never forces a colour change
        }
        self.lost = 0;
        if reading.colour == self.streak {
            self.streak_len += 1;
        } else {
            self.streak = reading.colour;
            self.streak_len = 1;
        }
        if self.streak_len >= self.needed && self.current != self.streak {
            self.current = self.streak;
            self.streak_len = 0;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb(rgb))
    }

    #[test]
    fn reads_the_three_lamp_colours() {
        assert_eq!(classify(&solid(20, 20, [230, 30, 30])).colour, LightColor::Red);
        assert_eq!(classify(&solid(20, 20, [255, 170, 0])).colour, LightColor::Amber);
        assert_eq!(classify(&solid(20, 20, [40, 200, 70])).colour, LightColor::Green);
    }

    #[test]
    fn grey_is_off_and_blue_is_not_a_lamp() {
        assert_eq!(classify(&solid(20, 20, [128, 128, 128])).colour, LightColor::Off);
        assert_eq!(classify(&solid(20, 20, [0, 0, 255])).colour, LightColor::Off);
    }

    #[test]
    fn a_lit_lamp_is_found_by_the_model_free_detector() {
        let mut img = solid(80, 60, [20, 22, 26]);
        for y in 25..35 {
            for x in 35..45 {
                img.put_pixel(x, y, Rgb([230, 40, 40]));
            }
        }
        let found = detect_lights(&img, &LightConfig::default());
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].reading.colour, LightColor::Red);
        let b = found[0].bbox;
        assert!((b.x1 - 35.0).abs() < 2.0 && (b.y1 - 25.0).abs() < 2.0);
    }

    #[test]
    fn the_tracker_needs_agreement_before_switching() {
        let red = LightReading { colour: LightColor::Red, confidence: 0.9 };
        let green = LightReading { colour: LightColor::Green, confidence: 0.9 };
        let mut tracker = LightTracker::new(3);
        assert!(!tracker.observe(red));
        assert!(!tracker.observe(red));
        assert!(tracker.observe(red));
        assert_eq!(tracker.current(), LightColor::Red);
        // One green frame must not flip the state.
        assert!(!tracker.observe(green));
        assert_eq!(tracker.current(), LightColor::Red);
        assert!(!tracker.observe(green));
        assert!(tracker.observe(green));
        assert_eq!(tracker.current(), LightColor::Green);
    }
}
