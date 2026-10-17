//! Drawing detection results onto camera frames.

use image::{GrayImage, RgbImage};
use tiny_skia::Pixmap;

use crate::lanes::{Departure, Lane, LaneFrame};
use crate::render::{self, rgba, Align, Fonts};

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
