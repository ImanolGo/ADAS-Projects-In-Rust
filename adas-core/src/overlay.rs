//! Drawing detection results onto camera frames.

use image::{GrayImage, RgbImage};
use tiny_skia::Pixmap;

use crate::detector::Detection;
use crate::lanes::{Departure, Lane, LaneFrame};
use crate::render::{self, rgba, Align, Fonts};
use crate::traffic_light::{LightCandidate, LightColor};

const GREEN: (u8, u8, u8) = (64, 210, 120);
const AMBER: (u8, u8, u8) = (255, 176, 0);

/// Shade the ego lane and draw both lines. Green normally, amber when drifting.
pub fn draw_lanes(pix: &mut Pixmap, lanes: &LaneFrame) {
    let (r, g, b) = if lanes.departure == Departure::None { GREEN } else { AMBER };
    if let (Some(l), Some(rt)) = (lanes.left, lanes.right) {
        let poly = [l.bottom, l.top, rt.top, rt.bottom];
        render::fill_polygon(pix, &poly, rgba(r, g, b, 0.22));
    }
    let line = |pix: &mut Pixmap, lane: Option<Lane>, departing: bool| {
        if let Some(lane) = lane {
            let colour = if departing { AMBER } else { (r, g, b) };
            render::polyline(
                pix,
                &[lane.bottom, lane.top],
                7.0,
                rgba(colour.0, colour.1, colour.2, 0.95),
                true,
            );
        }
    };
    line(pix, lanes.left, lanes.departure == Departure::Left);
    line(pix, lanes.right, lanes.departure == Departure::Right);
}

/// A small caption in the top-left corner, e.g. the lane offset.
pub fn caption(pix: &mut Pixmap, fonts: &Fonts, text: &str) {
    let size = (pix.height() as f32 * 0.04).max(14.0);
    let w = render::text_width(&fonts.semibold, text, size);
    render::fill_rounded_rect(pix, 14.0, 14.0, w + size, size * 1.7, size * 0.4, rgba(12, 14, 18, 0.7));
    render::text(
        pix,
        &fonts.semibold,
        text,
        14.0 + size * 0.5,
        14.0 + size * 1.2,
        size,
        rgba(238, 241, 243, 1.0),
        Align::Left,
    );
}

/// Paste a (small) edge map into the top-right corner: what the detector "sees".
pub fn edges_inset(frame: &mut RgbImage, edges: &GrayImage) {
    let scale = (frame.width() as f32 * 0.3 / edges.width() as f32).min(1.0);
    let (w, h) = ((edges.width() as f32 * scale) as u32, (edges.height() as f32 * scale) as u32);
    let small = image::imageops::resize(edges, w.max(1), h.max(1), image::imageops::FilterType::Triangle);
    let x0 = frame.width().saturating_sub(w + 14);
    for (x, y, p) in small.enumerate_pixels() {
        if let Some(dst) = frame.get_pixel_mut_checked(x0 + x, 14 + y) {
            let v = p[0];
            *dst =
                if v > 40 { image::Rgb([v, v, v]) } else { image::Rgb([dst[0] / 4, dst[1] / 4, dst[2] / 4]) };
        }
    }
}

/// A distinct colour per COCO class, so boxes are easy to tell apart.
pub fn class_colour(class_id: usize) -> (u8, u8, u8) {
    const PALETTE: [(u8, u8, u8); 6] = [
        (90, 200, 250), // cyan
        (255, 159, 10), // orange
        (48, 209, 88),  // green
        (191, 90, 242), // purple
        (255, 214, 10), // yellow
        (255, 69, 58),  // red
    ];
    PALETTE[class_id % PALETTE.len()]
}

/// Draw YOLOX boxes with class labels.
pub fn draw_detections(pix: &mut Pixmap, fonts: &Fonts, detections: &[Detection]) {
    for d in detections {
        let colour = class_colour(d.class_id);
        let b = d.bbox;
        render::stroke_rect(
            pix,
            b.x1,
            b.y1,
            b.width().max(2.0),
            b.height().max(2.0),
            3.0,
            rgba(colour.0, colour.1, colour.2, 0.95),
        );
        let size = (pix.height() as f32 * 0.028).max(12.0);
        let text = format!("{} {:.0}%", d.label(), d.confidence * 100.0);
        let tw = render::text_width(&fonts.semibold, &text, size);
        let chip_h = size * 1.5;
        let chip_y = (b.y1 - chip_h).max(0.0);
        render::fill_rounded_rect(
            pix,
            b.x1,
            chip_y,
            tw + size * 0.8,
            chip_h,
            size * 0.35,
            rgba(colour.0, colour.1, colour.2, 0.92),
        );
        render::text(
            pix,
            &fonts.semibold,
            &text,
            b.x1 + size * 0.4,
            chip_y + size * 1.08,
            size,
            rgba(16, 18, 20, 1.0),
            Align::Left,
        );
    }
}

/// Draw traffic-light boxes, coloured by the recognised lamp.
pub fn draw_lights(pix: &mut Pixmap, fonts: &Fonts, lights: &[LightCandidate]) {
    for light in lights {
        let colour = light.reading.colour.rgb();
        let b = light.bbox;
        render::stroke_rect(
            pix,
            b.x1,
            b.y1,
            b.width().max(2.0),
            b.height().max(2.0),
            3.0,
            rgba(colour.0, colour.1, colour.2, 0.95),
        );
        let size = (pix.height() as f32 * 0.028).max(12.0);
        let text = format!("{} {:.0}%", light.reading.colour.as_str(), light.reading.confidence * 100.0);
        let tw = render::text_width(&fonts.semibold, &text, size);
        let chip_h = size * 1.5;
        let chip_y = (b.y1 - chip_h).max(0.0);
        render::fill_rounded_rect(
            pix,
            b.x1,
            chip_y,
            tw + size * 0.8,
            chip_h,
            size * 0.35,
            rgba(colour.0, colour.1, colour.2, 0.92),
        );
        render::text(
            pix,
            &fonts.semibold,
            &text,
            b.x1 + size * 0.4,
            chip_y + size * 1.08,
            size,
            rgba(16, 18, 20, 1.0),
            Align::Left,
        );
    }
}

/// A three-lamp panel (top-right) showing the confirmed traffic-light state.
pub fn light_panel(pix: &mut Pixmap, fonts: &Fonts, colour: LightColor) {
    let r = 14.0;
    let pad = 10.0;
    let w = r * 2.0 + pad * 2.0;
    let h = r * 6.0 + pad * 4.0;
    let x = pix.width() as f32 - w - 16.0;
    let y = 16.0;
    render::fill_rounded_rect(pix, x, y, w, h, 10.0, rgba(12, 14, 18, 0.75));
    for (i, lamp) in [LightColor::Red, LightColor::Amber, LightColor::Green].iter().enumerate() {
        let cx = x + pad + r;
        let cy = y + pad + r + i as f32 * (r * 2.0 + pad);
        let (cr, cg, cb) = lamp.rgb();
        let on = *lamp == colour;
        let fill = if on { rgba(cr, cg, cb, 1.0) } else { rgba(cr / 5, cg / 5, cb / 5, 1.0) };
        render::fill_circle(pix, cx, cy, r, fill);
        if on {
            render::stroke_circle(pix, cx, cy, r, 2.0, rgba(238, 241, 243, 0.9));
        }
    }
    render::text(
        pix,
        &fonts.bold,
        colour.as_str(),
        x + w / 2.0,
        y + h + 18.0,
        17.0,
        rgba(238, 241, 243, 1.0),
        Align::Center,
    );
}
