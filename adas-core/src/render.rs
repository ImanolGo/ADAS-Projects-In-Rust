//! Drawing helpers: tiny-skia for antialiased shapes, ab_glyph for text, and
//! the German road signs used by the cluster and the drive demo.

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::{Rgb, RgbImage};
use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Rect, Stroke, Transform};

use crate::paths::{FONT_BOLD, FONT_SEMIBOLD};

/// The two compiled-in faces (Barlow Semi Condensed: DIN-like, like road signs).
pub struct Fonts {
    pub semibold: FontRef<'static>,
    pub bold: FontRef<'static>,
}

impl Fonts {
    pub fn load() -> Self {
        Self {
            semibold: FontRef::try_from_slice(FONT_SEMIBOLD).expect("bundled font is valid"),
            bold: FontRef::try_from_slice(FONT_BOLD).expect("bundled font is valid"),
        }
    }
}

impl Default for Fonts {
    fn default() -> Self {
        Self::load()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Align {
    Left,
    Center,
    Right,
}

pub fn rgba(r: u8, g: u8, b: u8, a: f32) -> Color {
    Color::from_rgba8(r, g, b, (a.clamp(0.0, 1.0) * 255.0).round() as u8)
}

// -- image <-> pixmap ---------------------------------------------------------

pub fn pixmap_from_rgb(img: &RgbImage) -> Pixmap {
    let mut pix = Pixmap::new(img.width(), img.height()).expect("non-empty image");
    for (dst, src) in pix.data_mut().chunks_exact_mut(4).zip(img.pixels()) {
        dst.copy_from_slice(&[src[0], src[1], src[2], 255]);
    }
    pix
}

/// Flatten onto black (frames are opaque, so alpha is always 255 in practice).
pub fn rgb_from_pixmap(pix: &Pixmap) -> RgbImage {
    let mut img = RgbImage::new(pix.width(), pix.height());
    for (dst, src) in img.pixels_mut().zip(pix.data().chunks_exact(4)) {
        *dst = Rgb([src[0], src[1], src[2]]);
    }
    img
}

// -- shapes -------------------------------------------------------------------

fn paint(color: Color) -> Paint<'static> {
    let mut p = Paint::default();
    p.set_color(color);
    p.anti_alias = true;
    p
}

pub fn fill_circle(pix: &mut Pixmap, cx: f32, cy: f32, r: f32, color: Color) {
    if let Some(path) = PathBuilder::from_circle(cx, cy, r.max(0.1)) {
        pix.fill_path(&path, &paint(color), FillRule::Winding, Transform::identity(), None);
    }
}

pub fn stroke_circle(pix: &mut Pixmap, cx: f32, cy: f32, r: f32, width: f32, color: Color) {
    if let Some(path) = PathBuilder::from_circle(cx, cy, r.max(0.1)) {
        let stroke = Stroke { width, ..Stroke::default() };
        pix.stroke_path(&path, &paint(color), &stroke, Transform::identity(), None);
    }
}

pub fn fill_rect(pix: &mut Pixmap, x: f32, y: f32, w: f32, h: f32, color: Color) {
    if let Some(rect) = Rect::from_xywh(x, y, w.max(0.1), h.max(0.1)) {
        pix.fill_rect(rect, &paint(color), Transform::identity(), None);
    }
}

/// Rounded rectangle; `radius` is clamped to half the shorter side.
pub fn fill_rounded_rect(pix: &mut Pixmap, x: f32, y: f32, w: f32, h: f32, radius: f32, color: Color) {
    let r = radius.min(w / 2.0).min(h / 2.0);
    let k = 0.552_284_8 * r; // circle-ish Bézier constant
    let mut pb = PathBuilder::new();
    pb.move_to(x + r, y);
    pb.line_to(x + w - r, y);
    pb.cubic_to(x + w - r + k, y, x + w, y + r - k, x + w, y + r);
    pb.line_to(x + w, y + h - r);
    pb.cubic_to(x + w, y + h - r + k, x + w - r + k, y + h, x + w - r, y + h);
    pb.line_to(x + r, y + h);
    pb.cubic_to(x + r - k, y + h, x, y + h - r + k, x, y + h - r);
    pb.line_to(x, y + r);
    pb.cubic_to(x, y + r - k, x + r - k, y, x + r, y);
    pb.close();
    if let Some(path) = pb.finish() {
        pix.fill_path(&path, &paint(color), FillRule::Winding, Transform::identity(), None);
    }
}

pub fn polyline(pix: &mut Pixmap, points: &[(f32, f32)], width: f32, color: Color, round: bool) {
    let Some((first, rest)) = points.split_first() else { return };
    let mut pb = PathBuilder::new();
    pb.move_to(first.0, first.1);
    for p in rest {
        pb.line_to(p.0, p.1);
    }
    if let Some(path) = pb.finish() {
        let mut stroke = Stroke { width, ..Stroke::default() };
        if round {
            stroke.line_cap = tiny_skia::LineCap::Round;
            stroke.line_join = tiny_skia::LineJoin::Round;
        }
        pix.stroke_path(&path, &paint(color), &stroke, Transform::identity(), None);
    }
}

pub fn fill_polygon(pix: &mut Pixmap, points: &[(f32, f32)], color: Color) {
    let Some((first, rest)) = points.split_first() else { return };
    let mut pb = PathBuilder::new();
    pb.move_to(first.0, first.1);
    for p in rest {
        pb.line_to(p.0, p.1);
    }
    pb.close();
    if let Some(path) = pb.finish() {
        pix.fill_path(&path, &paint(color), FillRule::Winding, Transform::identity(), None);
    }
}

/// An arc of a circle from `start` to `end` degrees (0° = east, clockwise, screen coordinates).
#[allow(clippy::too_many_arguments)] // drawing calls read best with plain coordinates
pub fn stroke_arc(
    pix: &mut Pixmap,
    cx: f32,
    cy: f32,
    r: f32,
    start: f32,
    end: f32,
    width: f32,
    color: Color,
) {
    if (end - start).abs() < 0.1 {
        return;
    }
    let steps = ((end - start).abs() / 3.0).ceil().max(2.0) as usize;
    let pts: Vec<(f32, f32)> = (0..=steps)
        .map(|i| {
            let a = (start + (end - start) * i as f32 / steps as f32).to_radians();
            (cx + r * a.cos(), cy + r * a.sin())
        })
        .collect();
    polyline(pix, &pts, width, color, true);
}

// -- text ---------------------------------------------------------------------

/// `size` is the em size in pixels (as in CSS or PIL). ab_glyph's `PxScale` is
/// the ascent-to-descent height instead, which made every glyph ~20% too small.
fn px_scale(font: &FontRef<'_>, size: f32) -> PxScale {
    font.pt_to_px_scale(size).unwrap_or(PxScale::from(size))
}

/// Width of `text` in pixels at em size `size`.
pub fn text_width(font: &FontRef<'_>, text: &str, size: f32) -> f32 {
    let scaled = font.as_scaled(px_scale(font, size));
    let mut width = 0.0;
    let mut prev = None;
    for ch in text.chars() {
        let id = scaled.glyph_id(ch);
        if let Some(p) = prev {
            width += scaled.kern(p, id);
        }
        width += scaled.h_advance(id);
        prev = Some(id);
    }
    width
}

/// Draw `text` with its baseline at `y`.
#[allow(clippy::too_many_arguments)] // drawing calls read best with plain coordinates
pub fn text(
    pix: &mut Pixmap,
    font: &FontRef<'_>,
    text: &str,
    x: f32,
    y: f32,
    size: f32,
    color: Color,
    align: Align,
) {
    let scale = px_scale(font, size);
    let scaled = font.as_scaled(scale);
    let width = text_width(font, text, size);
    let mut caret = match align {
        Align::Left => x,
        Align::Center => x - width / 2.0,
        Align::Right => x - width,
    };
    let (pw, ph) = (pix.width() as i32, pix.height() as i32);
    let c = color.to_color_u8();
    let mut prev = None;
    for ch in text.chars() {
        let id = scaled.glyph_id(ch);
        if let Some(p) = prev {
            caret += scaled.kern(p, id);
        }
        let glyph = id.with_scale_and_position(scale, ab_glyph::point(caret, y));
        caret += scaled.h_advance(id);
        prev = Some(id);
        let Some(outline) = font.outline_glyph(glyph) else { continue };
        let bounds = outline.px_bounds();
        let data = pix.data_mut();
        outline.draw(|gx, gy, coverage| {
            let px = bounds.min.x as i32 + gx as i32;
            let py = bounds.min.y as i32 + gy as i32;
            if px < 0 || py < 0 || px >= pw || py >= ph {
                return;
            }
            let a = coverage.clamp(0.0, 1.0) * c.alpha() as f32 / 255.0;
            let i = ((py * pw + px) * 4) as usize;
            let blend = |src: u8, dst: u8| (src as f32 * a + dst as f32 * (1.0 - a)).round() as u8;
            data[i] = blend(c.red(), data[i]);
            data[i + 1] = blend(c.green(), data[i + 1]);
            data[i + 2] = blend(c.blue(), data[i + 2]);
            data[i + 3] = (a * 255.0 + data[i + 3] as f32 * (1.0 - a)).round() as u8;
        });
    }
}

/// Draw `text` centred both ways on (cx, cy), using the glyphs' real ink bounds.
pub fn text_centered(
    pix: &mut Pixmap,
    font: &FontRef<'_>,
    s: &str,
    cx: f32,
    cy: f32,
    size: f32,
    color: Color,
) {
    let scale = px_scale(font, size);
    // Digits and capitals sit between the baseline and the cap height.
    let cap = font
        .outline_glyph(font.glyph_id('8').with_scale(scale))
        .map(|g| -g.px_bounds().min.y)
        .unwrap_or(size * 0.7);
    text(pix, font, s, cx, cy + cap / 2.0, size, color, Align::Center);
}

// -- road signs ---------------------------------------------------------------

pub const SIGN_RED: (u8, u8, u8) = (196, 30, 46);
pub const SIGN_WHITE: (u8, u8, u8) = (242, 242, 236);
pub const SIGN_INK: (u8, u8, u8) = (24, 24, 26);

/// Ratio of a sign crop's side to the sign's outer radius that matches GTSRB
/// crops (the sign fills ~72% of the crop). Used to cut crops for the classifier.
pub const CROP_PER_RADIUS: f32 = 2.0 / 0.72;

/// A German speed-limit sign (Zeichen 274): red ring, white field, black numbers.
///
/// The proportions were tuned against the bundled classifier: it reads 30, 70
/// and 80 reliably, 20 and 100 most of the time, and 120 almost never.
pub fn speed_sign(pix: &mut Pixmap, fonts: &Fonts, cx: f32, cy: f32, r: f32, kmh: u32) {
    let (rr, rg, rb) = SIGN_RED;
    let (wr, wg, wb) = SIGN_WHITE;
    let (ir, ig, ib) = SIGN_INK;
    fill_circle(pix, cx, cy, r, rgba(rr, rg, rb, 1.0));
    fill_circle(pix, cx, cy, r * 0.70, rgba(wr, wg, wb, 1.0));
    let digits = kmh.to_string();
    let size = r * 0.95 * if digits.len() > 2 { 0.78 } else { 1.0 };
    text_centered(pix, &fonts.semibold, &digits, cx, cy, size, rgba(ir, ig, ib, 1.0));
}

/// "End of speed limit" (Zeichen 278): grey numbers struck through by five black bands.
pub fn end_of_limit_sign(pix: &mut Pixmap, fonts: &Fonts, cx: f32, cy: f32, r: f32, kmh: u32) {
    let (wr, wg, wb) = SIGN_WHITE;
    fill_circle(pix, cx, cy, r, rgba(wr, wg, wb, 1.0));
    stroke_circle(pix, cx, cy, r - r * 0.03, r * 0.06, rgba(30, 30, 30, 1.0));
    text_centered(pix, &fonts.semibold, &kmh.to_string(), cx, cy, r * 0.95, rgba(120, 120, 120, 1.0));
    for k in -2..=2 {
        let o = k as f32 * r * 0.09;
        let (dx, dy) = (r * 0.62, r * 0.62);
        polyline(
            pix,
            &[(cx - dx + o, cy + dy + o), (cx + dx + o, cy - dy + o)],
            r * 0.035,
            rgba(20, 20, 20, 1.0),
            false,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixmap_round_trip_is_lossless_for_opaque_images() {
        let img = RgbImage::from_fn(5, 4, |x, y| Rgb([x as u8 * 40, y as u8 * 50, 7]));
        assert_eq!(rgb_from_pixmap(&pixmap_from_rgb(&img)), img);
    }

    #[test]
    fn text_draws_ink_where_expected() {
        let fonts = Fonts::load();
        let mut pix = Pixmap::new(200, 80).unwrap();
        pix.fill(Color::WHITE);
        text_centered(&mut pix, &fonts.bold, "70", 100.0, 40.0, 48.0, Color::BLACK);
        let img = rgb_from_pixmap(&pix);
        let dark: Vec<(u32, u32)> =
            img.enumerate_pixels().filter(|p| p.2[0] < 100).map(|p| (p.0, p.1)).collect();
        assert!(!dark.is_empty());
        let mean_y = dark.iter().map(|p| p.1 as f32).sum::<f32>() / dark.len() as f32;
        let mean_x = dark.iter().map(|p| p.0 as f32).sum::<f32>() / dark.len() as f32;
        assert!((mean_y - 40.0).abs() < 4.0 && (mean_x - 100.0).abs() < 6.0, "{mean_x},{mean_y}");
    }
}
