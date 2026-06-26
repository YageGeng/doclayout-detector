use super::postprocess::decode_box_detections;
use super::preprocess::{PP_DOCLAYOUT_V3_IMAGE_SIZE, resize_rgb_to_chw_f32};
use super::types::{PPDocLayoutV3Detection, PPDocLayoutV3OwnedOutputs};
use crate::error::LayoutError;
use crate::preprocess::validate_page_image;
use crate::types::PageImage;
#[cfg(not(all(target_family = "wasm", feature = "backend-webgpu")))]
use std::time::Instant;
use tracing::Level;

#[derive(Debug, Clone)]
pub struct PPDocLayoutV3Options {
    pub confidence_threshold: f32,
    pub image_size: u32,
}

impl Default for PPDocLayoutV3Options {
    /// Returns runtime defaults matching the PP-DocLayoutV3 model card contract.
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            image_size: PP_DOCLAYOUT_V3_IMAGE_SIZE,
        }
    }
}

pub trait PPDocLayoutV3Inference {
    /// Runs model inference for one preprocessed CHW page tensor.
    fn infer(&self, input: &[f32]) -> Result<PPDocLayoutV3OwnedOutputs, LayoutError>;
}

#[derive(Debug, Clone)]
pub struct PPDocLayoutV3Detector<M> {
    model: M,
    options: PPDocLayoutV3Options,
}

impl<M> PPDocLayoutV3Detector<M>
where
    M: PPDocLayoutV3Inference,
{
    /// Create a detector from a model implementation and runtime options.
    pub fn new(model: M, options: PPDocLayoutV3Options) -> Self {
        Self { model, options }
    }

    /// Detect layout boxes for one page and emit timing events for each model step.
    pub fn detect_page(
        &self,
        image: &PageImage<'_>,
    ) -> Result<Vec<PPDocLayoutV3Detection>, LayoutError> {
        let total_started = StepTimer::start();

        let validate_started = StepTimer::start();
        validate_page_image(image)?;
        tracing::event!(
            Level::INFO,
            step = "validate_input",
            duration_ms = validate_started.elapsed_ms(),
            width = image.width,
            height = image.height,
            dpi = image.dpi,
            "pp_doclayout model step completed"
        );

        if self.options.image_size != PP_DOCLAYOUT_V3_IMAGE_SIZE {
            return Err(LayoutError::UnsupportedImageSize {
                expected: PP_DOCLAYOUT_V3_IMAGE_SIZE,
                actual: self.options.image_size,
            });
        }

        let preprocess_started = StepTimer::start();
        let input = resize_rgb_to_chw_f32(image, self.options.image_size)?;
        tracing::event!(
            Level::INFO,
            step = "preprocess",
            duration_ms = preprocess_started.elapsed_ms(),
            input_values = input.len(),
            image_size = self.options.image_size,
            "pp_doclayout model step completed"
        );

        let inference_started = StepTimer::start();
        let outputs = self.model.infer(&input)?;
        tracing::event!(
            Level::INFO,
            step = "inference",
            duration_ms = inference_started.elapsed_ms(),
            "pp_doclayout model step completed"
        );

        let postprocess_started = StepTimer::start();
        let detections = decode_box_detections(
            &outputs.as_raw_outputs(),
            image,
            self.options.confidence_threshold,
        )?;
        tracing::event!(
            Level::INFO,
            step = "postprocess",
            duration_ms = postprocess_started.elapsed_ms(),
            detections = detections.len(),
            confidence_threshold = self.options.confidence_threshold,
            "pp_doclayout model step completed"
        );

        tracing::event!(
            Level::INFO,
            step = "total",
            duration_ms = total_started.elapsed_ms(),
            detections = detections.len(),
            "pp_doclayout model pipeline completed"
        );

        Ok(detections)
    }
}

#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
impl PPDocLayoutV3Detector<crate::model::EmbeddedModel> {
    /// Detect layout boxes asynchronously and emit timing events for browser inference.
    pub async fn detect_page_async(
        &self,
        image: &PageImage<'_>,
    ) -> Result<Vec<PPDocLayoutV3Detection>, LayoutError> {
        let total_started = StepTimer::start();

        let validate_started = StepTimer::start();
        validate_page_image(image)?;
        tracing::event!(
            Level::INFO,
            step = "validate_input",
            duration_ms = validate_started.elapsed_ms(),
            width = image.width,
            height = image.height,
            dpi = image.dpi,
            "pp_doclayout model step completed"
        );

        if self.options.image_size != PP_DOCLAYOUT_V3_IMAGE_SIZE {
            return Err(LayoutError::UnsupportedImageSize {
                expected: PP_DOCLAYOUT_V3_IMAGE_SIZE,
                actual: self.options.image_size,
            });
        }

        let preprocess_started = StepTimer::start();
        let input = resize_rgb_to_chw_f32(image, self.options.image_size)?;
        tracing::event!(
            Level::INFO,
            step = "preprocess",
            duration_ms = preprocess_started.elapsed_ms(),
            input_values = input.len(),
            image_size = self.options.image_size,
            "pp_doclayout model step completed"
        );

        let inference_started = StepTimer::start();
        let outputs = self.model.infer_async(&input).await?;
        tracing::event!(
            Level::INFO,
            step = "inference",
            duration_ms = inference_started.elapsed_ms(),
            "pp_doclayout model step completed"
        );

        let postprocess_started = StepTimer::start();
        let detections = decode_box_detections(
            &outputs.as_raw_outputs(),
            image,
            self.options.confidence_threshold,
        )?;
        tracing::event!(
            Level::INFO,
            step = "postprocess",
            duration_ms = postprocess_started.elapsed_ms(),
            detections = detections.len(),
            confidence_threshold = self.options.confidence_threshold,
            "pp_doclayout model step completed"
        );

        tracing::event!(
            Level::INFO,
            step = "total",
            duration_ms = total_started.elapsed_ms(),
            detections = detections.len(),
            "pp_doclayout model pipeline completed"
        );

        Ok(detections)
    }
}

#[cfg(not(all(target_family = "wasm", feature = "backend-webgpu")))]
#[derive(Debug, Clone)]
struct StepTimer {
    started: Instant,
}

#[cfg(not(all(target_family = "wasm", feature = "backend-webgpu")))]
impl StepTimer {
    /// Start a monotonic timer for native model step logging.
    fn start() -> Self {
        Self {
            started: Instant::now(),
        }
    }

    /// Return elapsed milliseconds for native model step logging.
    fn elapsed_ms(&self) -> f64 {
        self.started.elapsed().as_secs_f64() * 1000.0
    }
}

#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
#[derive(Debug, Clone)]
struct StepTimer {
    started_ms: f64,
}

#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
impl StepTimer {
    /// Start a browser-compatible timer for WebGPU model step logging.
    fn start() -> Self {
        Self {
            started_ms: js_sys::Date::now(),
        }
    }

    /// Return elapsed milliseconds without using unsupported wasm system time.
    fn elapsed_ms(&self) -> f64 {
        js_sys::Date::now() - self.started_ms
    }
}
