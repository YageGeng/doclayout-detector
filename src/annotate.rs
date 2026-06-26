use crate::pp_doclayout::PPDocLayoutV3Label;

/// One layout detection to draw on a rendered page image.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnnotatedDetection {
    pub label: PPDocLayoutV3Label,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl From<&crate::pp_doclayout::PPDocLayoutV3Detection> for AnnotatedDetection {
    fn from(value: &crate::pp_doclayout::PPDocLayoutV3Detection) -> Self {
        Self {
            label: value.label,
            confidence: value.confidence,
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
        }
    }
}

/// Draw layout detection boxes into an RGBA page image.
pub fn annotate_page_rgba(
    rgba: &mut [u8],
    image_width: u32,
    image_height: u32,
    page_width: f32,
    page_height: f32,
    detections: &[AnnotatedDetection],
) {
    if image_width == 0 || image_height == 0 || page_width <= 0.0 || page_height <= 0.0 {
        return;
    }

    let expected_len = image_width as usize * image_height as usize * 4;
    if rgba.len() != expected_len {
        return;
    }

    let scale_x = image_width as f32 / page_width;
    let scale_y = image_height as f32 / page_height;

    for detection in detections {
        let color = detection.label.debug_color_rgba();
        let left = (detection.x * scale_x).round() as i32;
        let top = (detection.y * scale_y).round() as i32;
        let right = ((detection.x + detection.width) * scale_x).round() as i32;
        let bottom = ((detection.y + detection.height) * scale_y).round() as i32;

        draw_rect_outline(
            Canvas {
                rgba,
                width: image_width as i32,
                height: image_height as i32,
            },
            Rect {
                left,
                top,
                right,
                bottom,
            },
            color,
        );
    }
}

struct Canvas<'a> {
    rgba: &'a mut [u8],
    width: i32,
    height: i32,
}

#[derive(Clone, Copy)]
struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

fn draw_rect_outline(canvas: Canvas<'_>, rect: Rect, color: [u8; 4]) {
    let Canvas {
        rgba,
        width: image_width,
        height: image_height,
    } = canvas;
    let Rect {
        left,
        top,
        right,
        bottom,
    } = rect;
    if right < left || bottom < top {
        return;
    }

    let left = left.clamp(0, image_width - 1);
    let top = top.clamp(0, image_height - 1);
    let right = right.clamp(0, image_width - 1);
    let bottom = bottom.clamp(0, image_height - 1);

    for thickness in 0..2 {
        draw_horizontal_line(
            rgba,
            image_width,
            image_height,
            left,
            right,
            top + thickness,
            color,
        );
        draw_horizontal_line(
            rgba,
            image_width,
            image_height,
            left,
            right,
            bottom - thickness,
            color,
        );
        draw_vertical_line(
            rgba,
            image_width,
            image_height,
            left + thickness,
            top,
            bottom,
            color,
        );
        draw_vertical_line(
            rgba,
            image_width,
            image_height,
            right - thickness,
            top,
            bottom,
            color,
        );
    }
}

fn draw_horizontal_line(
    rgba: &mut [u8],
    image_width: i32,
    image_height: i32,
    left: i32,
    right: i32,
    y: i32,
    color: [u8; 4],
) {
    if y < 0 || y >= image_height {
        return;
    }

    for x in left.max(0)..=right.min(image_width - 1) {
        put_pixel(rgba, image_width, x, y, color);
    }
}

fn draw_vertical_line(
    rgba: &mut [u8],
    image_width: i32,
    image_height: i32,
    x: i32,
    top: i32,
    bottom: i32,
    color: [u8; 4],
) {
    if x < 0 || x >= image_width {
        return;
    }

    for y in top.max(0)..=bottom.min(image_height - 1) {
        put_pixel(rgba, image_width, x, y, color);
    }
}

fn put_pixel(rgba: &mut [u8], image_width: i32, x: i32, y: i32, color: [u8; 4]) {
    let offset = ((y * image_width + x) * 4) as usize;
    rgba[offset..offset + 4].copy_from_slice(&color);
}
