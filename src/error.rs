use thiserror::Error;

/// Errors returned by layout preprocessing, inference, and decoding.
#[derive(Debug, Error)]
pub enum LayoutError {
    /// RGB input length does not match `width * height * 3`.
    #[error("invalid RGB buffer length: expected {expected}, got {actual}")]
    InvalidImageBuffer { expected: usize, actual: usize },
    /// The embedded model artifact was not generated into the build output.
    #[error("layout model is not available; rebuild with PP-DocLayoutV3 model assets")]
    ModelUnavailable,
    /// The caller requested an input size that the embedded model does not support.
    #[error("unsupported layout image size: expected {expected}, got {actual}")]
    UnsupportedImageSize { expected: u32, actual: u32 },
    /// The model output shape or tensor values could not be decoded.
    #[error("invalid layout output tensor: {0}")]
    InvalidModelOutput(String),
}
