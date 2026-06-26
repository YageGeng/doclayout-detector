use super::postprocess::decode_box_detections;
use super::preprocess::{PP_DOCLAYOUT_V3_IMAGE_SIZE, resize_rgb_to_chw_f32};
use super::types::{PPDocLayoutV3Detection, PPDocLayoutV3OwnedOutputs};
use crate::error::LayoutError;
use crate::preprocess::validate_page_image;
use crate::types::PageImage;

#[derive(Debug, Clone)]
pub struct PPDocLayoutV3Options {
    pub confidence_threshold: f32,
    pub image_size: u32,
}

impl Default for PPDocLayoutV3Options {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            image_size: PP_DOCLAYOUT_V3_IMAGE_SIZE,
        }
    }
}

pub trait PPDocLayoutV3Inference {
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
    pub fn new(model: M, options: PPDocLayoutV3Options) -> Self {
        Self { model, options }
    }

    pub fn detect_page(
        &self,
        image: &PageImage<'_>,
    ) -> Result<Vec<PPDocLayoutV3Detection>, LayoutError> {
        validate_page_image(image)?;
        if self.options.image_size != PP_DOCLAYOUT_V3_IMAGE_SIZE {
            return Err(LayoutError::UnsupportedImageSize {
                expected: PP_DOCLAYOUT_V3_IMAGE_SIZE,
                actual: self.options.image_size,
            });
        }

        let input = resize_rgb_to_chw_f32(image, self.options.image_size)?;
        let outputs = self.model.infer(&input)?;
        decode_box_detections(
            &outputs.as_raw_outputs(),
            image,
            self.options.confidence_threshold,
        )
    }
}
