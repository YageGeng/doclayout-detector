use crate::error::LayoutError;
use crate::preprocess::validate_page_image;
use crate::types::PageImage;

pub const PP_DOCLAYOUT_V3_IMAGE_SIZE: u32 = 800;

/// Resize RGB page pixels to PP-DocLayoutV3's fixed square CHW input.
///
/// The official processor uses `target_size: [800, 800]` with `keep_ratio:
/// false`, `rescale_factor: 1 / 255`, and no mean/std shift.
pub fn resize_rgb_to_chw_f32(
    image: &PageImage<'_>,
    target_size: u32,
) -> Result<Vec<f32>, LayoutError> {
    validate_page_image(image)?;
    let target_size = target_size as usize;
    let mut input = vec![0.0; 3 * target_size * target_size];

    let source_width = image.width as usize;
    let source_height = image.height as usize;
    let scale_x = image.width as f32 / target_size as f32;
    let scale_y = image.height as f32 / target_size as f32;

    for target_y in 0..target_size {
        let source_y = (target_y as f32 + 0.5) * scale_y - 0.5;
        let y0 = source_y.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(source_height.saturating_sub(1));
        let wy = (source_y - y0 as f32).clamp(0.0, 1.0);

        for target_x in 0..target_size {
            let source_x = (target_x as f32 + 0.5) * scale_x - 0.5;
            let x0 = source_x.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(source_width.saturating_sub(1));
            let wx = (source_x - x0 as f32).clamp(0.0, 1.0);
            let output_offset = target_y * target_size + target_x;

            for channel in 0..3 {
                let p00 = image.rgb[(y0 * source_width + x0) * 3 + channel] as f32;
                let p01 = image.rgb[(y0 * source_width + x1) * 3 + channel] as f32;
                let p10 = image.rgb[(y1 * source_width + x0) * 3 + channel] as f32;
                let p11 = image.rgb[(y1 * source_width + x1) * 3 + channel] as f32;
                let top = p00 * (1.0 - wx) + p01 * wx;
                let bottom = p10 * (1.0 - wx) + p11 * wx;
                input[channel * target_size * target_size + output_offset] =
                    (top * (1.0 - wy) + bottom * wy) / 255.0;
            }
        }
    }

    Ok(input)
}
