use crate::error::LayoutError;
use crate::types::PageImage;

/// Validate that a page image contains tightly packed RGB bytes.
///
/// # Errors
///
/// Returns [`LayoutError::InvalidImageBuffer`] when the byte length does not
/// match `width * height * 3`.
pub fn validate_page_image(image: &PageImage<'_>) -> Result<(), LayoutError> {
    let expected = image.width as usize * image.height as usize * 3;
    let actual = image.rgb.len();
    if actual != expected {
        return Err(LayoutError::InvalidImageBuffer { expected, actual });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_rgb_buffer_length() {
        let image = PageImage {
            rgb: &[0; 12],
            width: 2,
            height: 2,
            page_width: 2.0,
            page_height: 2.0,
            dpi: 72.0,
        };

        assert!(validate_page_image(&image).is_ok());
    }
}
