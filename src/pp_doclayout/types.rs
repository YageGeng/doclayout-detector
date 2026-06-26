use super::labels::PPDocLayoutV3Label;
use crate::error::LayoutError;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PPDocLayoutV3Detection {
    pub label: PPDocLayoutV3Label,
    pub confidence: f32,
    pub order: usize,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

pub struct PPDocLayoutV3RawOutputs<'a> {
    pub logits_shape: [usize; 3],
    pub logits: &'a [f32],
    pub pred_boxes_shape: [usize; 3],
    pub pred_boxes: &'a [f32],
    pub order_logits_shape: Option<[usize; 3]>,
    pub order_logits: Option<&'a [f32]>,
}

impl<'a> PPDocLayoutV3RawOutputs<'a> {
    /// Returns the batch dimension shared by all output tensors.
    pub fn batch_size(&self) -> usize {
        self.logits_shape[0]
    }

    /// Returns a borrowed single-page view for one batch index.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::InvalidModelOutput`] when the requested page is
    /// outside the output batch or when the tensor buffers cannot contain the
    /// advertised batch shape.
    pub fn page(&self, batch_index: usize) -> Result<Self, LayoutError> {
        if batch_index >= self.batch_size() {
            return Err(LayoutError::InvalidModelOutput(format!(
                "batch index {batch_index} is outside output batch {}",
                self.batch_size()
            )));
        }

        let class_count = self.logits_shape[2];
        let logits_page_len = self.logits_shape[1] * class_count;
        let boxes_page_len = self.pred_boxes_shape[1] * self.pred_boxes_shape[2];
        let logits_start = batch_index * logits_page_len;
        let boxes_start = batch_index * boxes_page_len;
        let logits = self
            .logits
            .get(logits_start..logits_start + logits_page_len)
            .ok_or_else(|| {
                LayoutError::InvalidModelOutput(format!(
                    "logits buffer cannot provide batch index {batch_index}"
                ))
            })?;
        let pred_boxes = self
            .pred_boxes
            .get(boxes_start..boxes_start + boxes_page_len)
            .ok_or_else(|| {
                LayoutError::InvalidModelOutput(format!(
                    "pred box buffer cannot provide batch index {batch_index}"
                ))
            })?;

        let (order_logits_shape, order_logits) = match (self.order_logits_shape, self.order_logits)
        {
            (Some(shape), Some(values)) => {
                let order_page_len = shape[1] * shape[2];
                let order_start = batch_index * order_page_len;
                let page_values = values
                    .get(order_start..order_start + order_page_len)
                    .ok_or_else(|| {
                        LayoutError::InvalidModelOutput(format!(
                            "order logits buffer cannot provide batch index {batch_index}"
                        ))
                    })?;
                (Some([1, shape[1], shape[2]]), Some(page_values))
            }
            (None, None) => (None, None),
            (shape, values) => {
                return Err(LayoutError::InvalidModelOutput(format!(
                    "expected matching order logits shape and values, got shape {shape:?} and {} values",
                    values.map_or(0, <[f32]>::len)
                )));
            }
        };

        Ok(Self {
            logits_shape: [1, self.logits_shape[1], class_count],
            logits,
            pred_boxes_shape: [1, self.pred_boxes_shape[1], self.pred_boxes_shape[2]],
            pred_boxes,
            order_logits_shape,
            order_logits,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PPDocLayoutV3OwnedOutputs {
    pub logits_shape: [usize; 3],
    pub logits: Vec<f32>,
    pub pred_boxes_shape: [usize; 3],
    pub pred_boxes: Vec<f32>,
    pub order_logits_shape: Option<[usize; 3]>,
    pub order_logits: Option<Vec<f32>>,
}

impl PPDocLayoutV3OwnedOutputs {
    /// Borrows owned model outputs as raw slices for postprocessing.
    pub fn as_raw_outputs(&self) -> PPDocLayoutV3RawOutputs<'_> {
        PPDocLayoutV3RawOutputs {
            logits_shape: self.logits_shape,
            logits: &self.logits,
            pred_boxes_shape: self.pred_boxes_shape,
            pred_boxes: &self.pred_boxes,
            order_logits_shape: self.order_logits_shape,
            order_logits: self.order_logits.as_deref(),
        }
    }

    /// Concatenates single-page outputs into one batched output object.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::InvalidModelOutput`] when no pages are provided
    /// or when any page output does not use the expected single-page shape.
    pub fn from_single_page_outputs(pages: Vec<Self>) -> Result<Self, LayoutError> {
        let batch_size = pages.len();
        if batch_size == 0 {
            return Err(LayoutError::InvalidModelOutput(
                "cannot build batched outputs from zero pages".to_string(),
            ));
        }

        let logits_shape = pages[0].logits_shape;
        let pred_boxes_shape = pages[0].pred_boxes_shape;
        let order_logits_shape = pages[0].order_logits_shape;
        let mut logits = Vec::with_capacity(pages.iter().map(|page| page.logits.len()).sum());
        let mut pred_boxes =
            Vec::with_capacity(pages.iter().map(|page| page.pred_boxes.len()).sum());
        let mut order_logits = order_logits_shape.map(|_| {
            Vec::with_capacity(
                pages
                    .iter()
                    .filter_map(|page| page.order_logits.as_ref())
                    .map(Vec::len)
                    .sum(),
            )
        });

        for page in pages {
            if page.logits_shape[0] != 1
                || page.pred_boxes_shape[0] != 1
                || page.logits_shape[1..] != logits_shape[1..]
                || page.pred_boxes_shape[1..] != pred_boxes_shape[1..]
                || page.order_logits_shape != order_logits_shape
            {
                return Err(LayoutError::InvalidModelOutput(
                    "single-page outputs have incompatible shapes".to_string(),
                ));
            }

            logits.extend(page.logits);
            pred_boxes.extend(page.pred_boxes);
            if let Some(target) = order_logits.as_mut() {
                let Some(values) = page.order_logits else {
                    return Err(LayoutError::InvalidModelOutput(
                        "missing order logits for one batched page".to_string(),
                    ));
                };
                target.extend(values);
            }
        }

        Ok(Self {
            logits_shape: [batch_size, logits_shape[1], logits_shape[2]],
            logits,
            pred_boxes_shape: [batch_size, pred_boxes_shape[1], pred_boxes_shape[2]],
            pred_boxes,
            order_logits_shape: order_logits_shape.map(|shape| [batch_size, shape[1], shape[2]]),
            order_logits,
        })
    }
}

impl From<PPDocLayoutV3Detection> for crate::types::LayoutDetection {
    /// Converts PP-DocLayout detections into the crate-level layout detection type.
    fn from(value: PPDocLayoutV3Detection) -> Self {
        Self {
            label: value.label,
            confidence: value.confidence,
            order: value.order,
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
        }
    }
}
