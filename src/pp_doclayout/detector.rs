use super::postprocess::{decode_box_detections, decode_box_detections_batch};
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

    /// Runs model inference for a preprocessed batch of CHW page tensors.
    fn infer_batch(
        &self,
        input: &[f32],
        batch_size: usize,
    ) -> Result<PPDocLayoutV3OwnedOutputs, LayoutError> {
        if batch_size == 0 {
            return Err(LayoutError::InvalidModelOutput(
                "batch size must be greater than zero".to_string(),
            ));
        }

        let page_len =
            3 * PP_DOCLAYOUT_V3_IMAGE_SIZE as usize * PP_DOCLAYOUT_V3_IMAGE_SIZE as usize;
        let expected = batch_size * page_len;
        if input.len() != expected {
            return Err(LayoutError::InvalidModelOutput(format!(
                "expected batched CHW input length {expected}, got {}",
                input.len()
            )));
        }

        let pages = input
            .chunks_exact(page_len)
            .map(|page| self.infer(page))
            .collect::<Result<Vec<_>, _>>()?;
        PPDocLayoutV3OwnedOutputs::from_single_page_outputs(pages)
    }
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

    /// Detect layout boxes for a batch of pages using one model inference call when supported.
    pub fn detect_pages(
        &self,
        images: &[PageImage<'_>],
    ) -> Result<Vec<Vec<PPDocLayoutV3Detection>>, LayoutError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let total_started = StepTimer::start();
        let validate_started = StepTimer::start();
        for image in images {
            validate_page_image(image)?;
        }
        tracing::event!(
            Level::INFO,
            step = "validate_input_batch",
            duration_ms = validate_started.elapsed_ms(),
            pages = images.len(),
            "pp_doclayout model batch step completed"
        );

        if self.options.image_size != PP_DOCLAYOUT_V3_IMAGE_SIZE {
            return Err(LayoutError::UnsupportedImageSize {
                expected: PP_DOCLAYOUT_V3_IMAGE_SIZE,
                actual: self.options.image_size,
            });
        }

        let preprocess_started = StepTimer::start();
        let page_len = 3 * self.options.image_size as usize * self.options.image_size as usize;
        let mut input = Vec::with_capacity(images.len() * page_len);
        for image in images {
            input.extend(resize_rgb_to_chw_f32(image, self.options.image_size)?);
        }
        tracing::event!(
            Level::INFO,
            step = "preprocess_batch",
            duration_ms = preprocess_started.elapsed_ms(),
            pages = images.len(),
            input_values = input.len(),
            image_size = self.options.image_size,
            "pp_doclayout model batch step completed"
        );

        let inference_started = StepTimer::start();
        let outputs = self.model.infer_batch(&input, images.len())?;
        tracing::event!(
            Level::INFO,
            step = "inference_batch",
            duration_ms = inference_started.elapsed_ms(),
            pages = images.len(),
            "pp_doclayout model batch step completed"
        );

        let postprocess_started = StepTimer::start();
        let detections = decode_box_detections_batch(
            &outputs.as_raw_outputs(),
            images,
            self.options.confidence_threshold,
        )?;
        tracing::event!(
            Level::INFO,
            step = "postprocess_batch",
            duration_ms = postprocess_started.elapsed_ms(),
            pages = images.len(),
            detections = detections.iter().map(Vec::len).sum::<usize>(),
            confidence_threshold = self.options.confidence_threshold,
            "pp_doclayout model batch step completed"
        );

        tracing::event!(
            Level::INFO,
            step = "total_batch",
            duration_ms = total_started.elapsed_ms(),
            pages = images.len(),
            detections = detections.iter().map(Vec::len).sum::<usize>(),
            "pp_doclayout model batch pipeline completed"
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

    /// Detect layout boxes asynchronously for a batch of pages in browser WebGPU.
    pub async fn detect_pages_async(
        &self,
        images: &[PageImage<'_>],
    ) -> Result<Vec<Vec<PPDocLayoutV3Detection>>, LayoutError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let total_started = StepTimer::start();
        let validate_started = StepTimer::start();
        for image in images {
            validate_page_image(image)?;
        }
        tracing::event!(
            Level::INFO,
            step = "validate_input_batch",
            duration_ms = validate_started.elapsed_ms(),
            pages = images.len(),
            "pp_doclayout model batch step completed"
        );

        if self.options.image_size != PP_DOCLAYOUT_V3_IMAGE_SIZE {
            return Err(LayoutError::UnsupportedImageSize {
                expected: PP_DOCLAYOUT_V3_IMAGE_SIZE,
                actual: self.options.image_size,
            });
        }

        let preprocess_started = StepTimer::start();
        let page_len = 3 * self.options.image_size as usize * self.options.image_size as usize;
        let mut input = Vec::with_capacity(images.len() * page_len);
        for image in images {
            input.extend(resize_rgb_to_chw_f32(image, self.options.image_size)?);
        }
        tracing::event!(
            Level::INFO,
            step = "preprocess_batch",
            duration_ms = preprocess_started.elapsed_ms(),
            pages = images.len(),
            input_values = input.len(),
            image_size = self.options.image_size,
            "pp_doclayout model batch step completed"
        );

        let inference_started = StepTimer::start();
        let outputs = self.model.infer_batch_async(&input, images.len()).await?;
        tracing::event!(
            Level::INFO,
            step = "inference_batch",
            duration_ms = inference_started.elapsed_ms(),
            pages = images.len(),
            "pp_doclayout model batch step completed"
        );

        let postprocess_started = StepTimer::start();
        let detections = decode_box_detections_batch(
            &outputs.as_raw_outputs(),
            images,
            self.options.confidence_threshold,
        )?;
        tracing::event!(
            Level::INFO,
            step = "postprocess_batch",
            duration_ms = postprocess_started.elapsed_ms(),
            pages = images.len(),
            detections = detections.iter().map(Vec::len).sum::<usize>(),
            confidence_threshold = self.options.confidence_threshold,
            "pp_doclayout model batch step completed"
        );

        tracing::event!(
            Level::INFO,
            step = "total_batch",
            duration_ms = total_started.elapsed_ms(),
            pages = images.len(),
            detections = detections.iter().map(Vec::len).sum::<usize>(),
            "pp_doclayout model batch pipeline completed"
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

#[cfg(all(target_family = "wasm", feature = "backend-webgpu", feature = "wasm"))]
#[derive(Debug, Clone)]
struct StepTimer {
    started_ms: f64,
}

#[cfg(all(target_family = "wasm", feature = "backend-webgpu", feature = "wasm"))]
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

#[cfg(all(
    target_family = "wasm",
    feature = "backend-webgpu",
    not(feature = "wasm")
))]
#[derive(Debug, Clone)]
struct StepTimer;

#[cfg(all(
    target_family = "wasm",
    feature = "backend-webgpu",
    not(feature = "wasm")
))]
impl StepTimer {
    /// Start a no-op timer when browser bindings are not enabled.
    fn start() -> Self {
        Self
    }

    /// Return zero elapsed time without pulling browser JS dependencies.
    fn elapsed_ms(&self) -> f64 {
        0.0
    }
}
