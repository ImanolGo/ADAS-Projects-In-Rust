//! A digital instrument cluster (Activity 6): speed, the recognised speed
//! limit, lane assist, and what the camera currently believes it sees.
//!
//! Colours follow ISO 2575 telltale conventions: green = system active,
//! amber = caution, red = warning. The speed-limit sign is the hero.

use tiny_skia::{Color, Pixmap};

use crate::lanes::Departure;
use crate::render::{self, rgba, Align, Fonts};

/// Everything the cluster shows for one frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClusterState {
    pub speed_kmh: f32,
    /// The limit in force (from the speed-limit tracker).
    pub limit: Option<u32>,
    /// When the limit last changed (seconds), for the "new sign" pop.
    pub limit_changed_at: Option<f32>,
    pub lanes: LaneStatus,
    /// The classifier's raw opinion this frame: (label, confidence).
    pub camera: Option<(String, f32)>,
    pub time_s: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LaneStatus {
    pub left: bool,
    pub right: bool,
    pub departure: Departure,
}

impl Default for LaneStatus {
    fn default() -> Self {
        Self { left: false, right: false, departure: Departure::None }
    }
}

/// How the current speed relates to the limit.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SpeedBand {
    Ok,
    /// A little over (amber).
    Over,
    /// Well over (red).
    WayOver,
}

pub fn speed_band(speed_kmh: f32, limit: Option<u32>) -> SpeedBand {
    match limit {
        Some(limit) if speed_kmh > limit as f32 + 10.0 => SpeedBand::WayOver,
        Some(limit) if speed_kmh > limit as f32 + 2.0 => SpeedBand::Over,
        _ => SpeedBand::Ok,
    }
}

const TEXT: (u8, u8, u8) = (238, 241, 243);
const DIM: (u8, u8, u8) = (138, 146, 156);
const TRACK: (u8, u8, u8) = (52, 57, 64);
const GREEN: (u8, u8, u8) = (64, 210, 120);
const AMBER: (u8, u8, u8) = (255, 176, 0);
const RED: (u8, u8, u8) = (255, 69, 58);

fn c((r, g, b): (u8, u8, u8), a: f32) -> Color {
    rgba(r, g, b, a)
}

/// Max speed on the dial.
pub const DIAL_MAX: f32 = 160.0;
const DIAL_START: f32 = 135.0; // degrees, screen coordinates (clockwise from east)
const DIAL_SWEEP: f32 = 270.0;

/// Draw the cluster into the rectangle (x, y, w, h) of `pix`.
pub fn draw_cluster(pix: &mut Pixmap, fonts: &Fonts, x: f32, y: f32, w: f32, h: f32, state: &ClusterState) {
    let t = state.time_s;
    let band = speed_band(state.speed_kmh, state.limit);
    let band_colour = match band {
        SpeedBand::Ok => TEXT,
        SpeedBand::Over => AMBER,
        SpeedBand::WayOver => RED,
    };

    // Glass panel.
    render::fill_rounded_rect(pix, x, y, w, h, h * 0.14, rgba(12, 14, 18, 0.86));
    render::fill_rect(pix, x + h * 0.14, y, w - h * 0.28, 1.0, rgba(255, 255, 255, 0.10));

    // -- speedometer (left) ----------------------------------------------------
    let (gx, gy, gr) = (x + (w * 0.16).max(h * 0.62), y + h * 0.54, h * 0.4);
    render::stroke_arc(pix, gx, gy, gr, DIAL_START, DIAL_START + DIAL_SWEEP, h * 0.045, c(TRACK, 1.0));
    let frac = (state.speed_kmh / DIAL_MAX).clamp(0.0, 1.0);
    let fill = match band {
        SpeedBand::Ok => (196, 214, 230),
        other => {
            if other == SpeedBand::Over {
                AMBER
            } else {
                RED
            }
        }
    };
    render::stroke_arc(pix, gx, gy, gr, DIAL_START, DIAL_START + DIAL_SWEEP * frac, h * 0.045, c(fill, 1.0));
    if let Some(limit) = state.limit {
        // A tick on the dial where the limit is.
        let a = (DIAL_START + DIAL_SWEEP * (limit as f32 / DIAL_MAX).min(1.0)).to_radians();
        let (r0, r1) = (gr - h * 0.06, gr + h * 0.06);
        render::polyline(
            pix,
            &[(gx + r0 * a.cos(), gy + r0 * a.sin()), (gx + r1 * a.cos(), gy + r1 * a.sin())],
            h * 0.025,
            c(render::SIGN_RED, 1.0),
            true,
        );
    }
    let speed_txt = format!("{:.0}", state.speed_kmh.max(0.0));
    render::text_centered(pix, &fonts.bold, &speed_txt, gx, gy - h * 0.04, h * 0.3, c(band_colour, 1.0));
    render::text(pix, &fonts.semibold, "km/h", gx, gy + h * 0.24, h * 0.1, c(DIM, 1.0), Align::Center);

    // -- speed limit (centre, the hero) ---------------------------------------
    let (sx, sy) = (x + w * 0.5, y + h * 0.47);
    let base_r = h * 0.33;
    let pop = state
        .limit_changed_at
        .map(|t0| t - t0)
        .filter(|dt| (0.0..0.6).contains(dt))
        .map(|dt| 1.0 + 0.18 * (1.0 - dt / 0.6) * (dt * 18.0).sin().abs())
        .unwrap_or(1.0);
    let r = base_r * pop;
    if band != SpeedBand::Ok {
        // Pulsing halo: "you are over the limit".
        let pulse = 0.35 + 0.35 * (t * 7.0).sin().abs();
        let colour = if band == SpeedBand::WayOver { RED } else { AMBER };
        render::stroke_circle(pix, sx, sy, r * 1.12, h * 0.035, c(colour, pulse));
    }
    match state.limit {
        Some(kmh) => render::speed_sign(pix, fonts, sx, sy, r, kmh),
        None => {
            render::stroke_circle(pix, sx, sy, r * 0.88, h * 0.03, c(TRACK, 1.0));
            render::text_centered(pix, &fonts.semibold, "––", sx, sy, r * 0.8, c(DIM, 1.0));
        }
    }
    let caption = match (state.limit, band) {
        (None, _) => "No limit detected".to_string(),
        (Some(_), SpeedBand::Ok) => "Speed limit".to_string(),
        (Some(limit), _) => format!("{:.0} km/h over", state.speed_kmh - limit as f32),
    };
    let caption_colour = match band {
        SpeedBand::Ok => DIM,
        SpeedBand::Over => AMBER,
        SpeedBand::WayOver => RED,
    };
    render::text(
        pix,
        &fonts.semibold,
        &caption,
        sx,
        y + h * 0.94,
        h * 0.11,
        c(caption_colour, 1.0),
        Align::Center,
    );

    // -- lane assist (right) ---------------------------------------------------
    let (lx, ly) = (x + w - (w * 0.16).max(h * 0.95), y + h * 0.42);
    let blink_on = (t * 4.0).fract() < 0.6;
    let line_colour = |tracked: bool, departing: bool| -> Color {
        if departing {
            c(AMBER, if blink_on { 1.0 } else { 0.25 })
        } else if tracked {
            c(GREEN, 1.0)
        } else {
            c(TRACK, 1.0)
        }
    };
    let (half_top, half_bottom, top, bottom) = (h * 0.08, h * 0.30, ly - h * 0.24, ly + h * 0.22);
    let lw = h * 0.035;
    render::polyline(
        pix,
        &[(lx - half_bottom, bottom), (lx - half_top, top)],
        lw,
        line_colour(state.lanes.left, state.lanes.departure == Departure::Left),
        true,
    );
    render::polyline(
        pix,
        &[(lx + half_bottom, bottom), (lx + half_top, top)],
        lw,
        line_colour(state.lanes.right, state.lanes.departure == Departure::Right),
        true,
    );
    // The car, nudged towards the side it is drifting to.
    let nudge = match state.lanes.departure {
        Departure::Left => -h * 0.07,
        Departure::Right => h * 0.07,
        Departure::None => 0.0,
    };
    render::fill_rounded_rect(
        pix,
        lx - h * 0.075 + nudge,
        ly - h * 0.02,
        h * 0.15,
        h * 0.2,
        h * 0.04,
        c(TEXT, 0.9),
    );
    render::fill_rect(pix, lx - h * 0.05 + nudge, ly + h * 0.01, h * 0.1, h * 0.045, rgba(12, 14, 18, 0.9));
    let lane_caption = match state.lanes.departure {
        Departure::Left => ("Drifting left", AMBER),
        Departure::Right => ("Drifting right", AMBER),
        Departure::None if state.lanes.left && state.lanes.right => ("Lane assist", GREEN),
        Departure::None => ("Looking for lanes", DIM),
    };
    render::text(
        pix,
        &fonts.semibold,
        lane_caption.0,
        lx,
        y + h * 0.80,
        h * 0.11,
        c(lane_caption.1, 1.0),
        Align::Center,
    );

    // What the camera believes right now (the raw classifier, before voting).
    let camera = match &state.camera {
        Some((label, conf)) => format!("Camera: {label} · {:.0}%", conf * 100.0),
        None => "Camera: no sign in view".to_string(),
    };
    render::text(pix, &fonts.semibold, &camera, lx, y + h * 0.94, h * 0.085, c(DIM, 1.0), Align::Center);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_bands() {
        assert_eq!(speed_band(80.0, None), SpeedBand::Ok);
        assert_eq!(speed_band(52.0, Some(50)), SpeedBand::Ok);
        assert_eq!(speed_band(55.0, Some(50)), SpeedBand::Over);
        assert_eq!(speed_band(65.0, Some(50)), SpeedBand::WayOver);
    }

    #[test]
    fn draws_without_panicking_in_every_state() {
        let fonts = Fonts::load();
        let mut pix = Pixmap::new(960, 220).unwrap();
        for (speed, limit, dep) in [
            (0.0, None, Departure::None),
            (64.0, Some(50), Departure::Left),
            (170.0, Some(120), Departure::Right),
        ] {
            let state = ClusterState {
                speed_kmh: speed,
                limit,
                limit_changed_at: Some(0.1),
                lanes: LaneStatus { left: true, right: false, departure: dep },
                camera: Some(("Speed limit (50km/h)".into(), 0.97)),
                time_s: 0.3,
            };
            draw_cluster(&mut pix, &fonts, 0.0, 0.0, 960.0, 220.0, &state);
        }
    }
}
