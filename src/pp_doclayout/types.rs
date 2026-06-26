use super::labels::PPDocLayoutV3Label;
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
