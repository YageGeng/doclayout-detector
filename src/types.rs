use crate::pp_doclayout::PPDocLayoutV3Label;
use serde::Serialize;

/// Borrowed page image input and corresponding PDF page dimensions.
///
/// The RGB buffer must be tightly packed as `width * height * 3` bytes. Page
/// dimensions are in PDF points and are used when detections are restored from
/// rendered pixels back into parser coordinates.
#[derive(Debug, Clone, Copy)]
pub struct PageImage<'a> {
    /// Tightly packed RGB pixels in row-major order.
    pub rgb: &'a [u8],
    /// Rendered image width in pixels.
    pub width: u32,
    /// Rendered image height in pixels.
    pub height: u32,
    /// PDF page width in points.
    pub page_width: f32,
    /// PDF page height in points.
    pub page_height: f32,
    /// Render DPI used to map pixels back into page coordinates.
    pub dpi: f32,
}

/// Page-space layout detection returned by the document layout detector.
///
/// Coordinates use the same top-left-origin PDF point space as LiteParse text
/// items, making detections directly comparable with extracted text boxes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayoutDetection {
    /// Stable layout class predicted by the model.
    pub label: PPDocLayoutV3Label,
    /// Model confidence score for the predicted label.
    pub confidence: f32,
    /// Reading-order rank derived from the model's pairwise order logits.
    pub order: usize,
    /// Left position in PDF page coordinates.
    pub x: f32,
    /// Top position in PDF page coordinates.
    pub y: f32,
    /// Detection width in PDF page coordinates.
    pub width: f32,
    /// Detection height in PDF page coordinates.
    pub height: f32,
}
