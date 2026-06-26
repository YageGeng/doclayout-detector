use super::labels::PPDocLayoutV3Label;
use super::types::{PPDocLayoutV3Detection, PPDocLayoutV3RawOutputs};
use crate::error::LayoutError;
use crate::types::PageImage;

const NUM_QUERIES: usize = 300;

/// Decodes raw PP-DocLayoutV3 tensors into page-space detections sorted by reading order.
pub fn decode_box_detections(
    outputs: &PPDocLayoutV3RawOutputs<'_>,
    image: &PageImage<'_>,
    threshold: f32,
) -> Result<Vec<PPDocLayoutV3Detection>, LayoutError> {
    let mut pages = decode_box_detections_batch(outputs, std::slice::from_ref(image), threshold)?;
    Ok(pages.remove(0))
}

/// Decodes batched PP-DocLayoutV3 tensors into per-page detections.
pub fn decode_box_detections_batch(
    outputs: &PPDocLayoutV3RawOutputs<'_>,
    images: &[PageImage<'_>],
    threshold: f32,
) -> Result<Vec<Vec<PPDocLayoutV3Detection>>, LayoutError> {
    validate_shapes(outputs)?;
    if outputs.batch_size() != images.len() {
        return Err(LayoutError::InvalidModelOutput(format!(
            "expected {} page images for output batch {}, got {}",
            outputs.batch_size(),
            outputs.batch_size(),
            images.len()
        )));
    }

    images
        .iter()
        .enumerate()
        .map(|(batch_index, image)| {
            decode_box_detections_for_page(&outputs.page(batch_index)?, image, threshold)
        })
        .collect()
}

/// Decodes one already-sliced output page into page-space detections.
fn decode_box_detections_for_page(
    outputs: &PPDocLayoutV3RawOutputs<'_>,
    image: &PageImage<'_>,
    threshold: f32,
) -> Result<Vec<PPDocLayoutV3Detection>, LayoutError> {
    let order_seq = outputs
        .order_logits
        .map(compute_order_sequence)
        .unwrap_or_else(|| (0..NUM_QUERIES).collect());

    let mut ranked = Vec::with_capacity(NUM_QUERIES * PPDocLayoutV3Label::class_count());
    for query_id in 0..NUM_QUERIES {
        let row_start = query_id * PPDocLayoutV3Label::class_count();
        for class_id in 0..PPDocLayoutV3Label::class_count() {
            let score = sigmoid(outputs.logits[row_start + class_id]);
            ranked.push((score, query_id, class_id));
        }
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    if std::env::var_os("LITEPARSE_LAYOUT_DEBUG").is_some() {
        let top = ranked
            .iter()
            .take(5)
            .map(|(score, query_id, class_id)| format!("q{query_id}/c{class_id}:{score:.4}"))
            .collect::<Vec<_>>()
            .join(", ");
        tracing::debug!(%top, threshold, "top raw detection scores");
    }

    let mut detections = Vec::new();
    let mut rejected_below_threshold = 0usize;
    let mut rejected_invalid_box = 0usize;
    for (confidence, query_id, class_id) in ranked.into_iter().take(NUM_QUERIES) {
        if confidence < threshold {
            rejected_below_threshold += 1;
            continue;
        }
        let Ok(label) = PPDocLayoutV3Label::try_from(class_id) else {
            continue;
        };
        let box_offset = query_id * 4;
        let center_x = outputs.pred_boxes[box_offset];
        let center_y = outputs.pred_boxes[box_offset + 1];
        let width = outputs.pred_boxes[box_offset + 2].max(0.0);
        let height = outputs.pred_boxes[box_offset + 3].max(0.0);
        if width <= 0.0 || height <= 0.0 {
            rejected_invalid_box += 1;
            continue;
        }

        let left = (center_x - width * 0.5).clamp(0.0, 1.0);
        let top = (center_y - height * 0.5).clamp(0.0, 1.0);
        let right = (center_x + width * 0.5).clamp(0.0, 1.0);
        let bottom = (center_y + height * 0.5).clamp(0.0, 1.0);
        let page_x = left * image.page_width;
        let page_y = top * image.page_height;
        let page_width = (right - left).max(0.0) * image.page_width;
        let page_height = (bottom - top).max(0.0) * image.page_height;
        if page_width <= 0.0 || page_height <= 0.0 {
            rejected_invalid_box += 1;
            continue;
        }

        detections.push(PPDocLayoutV3Detection {
            label,
            confidence,
            order: order_seq[query_id],
            x: page_x,
            y: page_y,
            width: page_width,
            height: page_height,
        });
    }

    if std::env::var_os("LITEPARSE_LAYOUT_DEBUG").is_some() {
        tracing::debug!(
            boxes = detections.len(),
            rejected_below_threshold,
            rejected_invalid_box,
            "decoded box detections"
        );
    }

    detections.sort_by(|a, b| {
        a.order
            .cmp(&b.order)
            .then_with(|| b.confidence.total_cmp(&a.confidence))
    });
    Ok(detections)
}

/// Verifies that batched model output tensors match the checkpoint's fixed per-page shape.
fn validate_shapes(outputs: &PPDocLayoutV3RawOutputs<'_>) -> Result<(), LayoutError> {
    let class_count = PPDocLayoutV3Label::class_count();
    let batch_size = outputs.batch_size();
    if batch_size == 0 {
        return Err(LayoutError::InvalidModelOutput(
            "output batch size must be greater than zero".to_string(),
        ));
    }
    if outputs.logits_shape != [batch_size, NUM_QUERIES, class_count] {
        return Err(LayoutError::InvalidModelOutput(format!(
            "expected logits [{batch_size}, {NUM_QUERIES}, {class_count}], got {:?}",
            outputs.logits_shape
        )));
    }
    if outputs.logits.len() != batch_size * NUM_QUERIES * class_count {
        return Err(LayoutError::InvalidModelOutput(format!(
            "expected {} logits, got {}",
            batch_size * NUM_QUERIES * class_count,
            outputs.logits.len()
        )));
    }
    if outputs.pred_boxes_shape != [batch_size, NUM_QUERIES, 4] {
        return Err(LayoutError::InvalidModelOutput(format!(
            "expected pred_boxes [{batch_size}, {NUM_QUERIES}, 4], got {:?}",
            outputs.pred_boxes_shape
        )));
    }
    if outputs.pred_boxes.len() != batch_size * NUM_QUERIES * 4 {
        return Err(LayoutError::InvalidModelOutput(format!(
            "expected {} pred box values, got {}",
            batch_size * NUM_QUERIES * 4,
            outputs.pred_boxes.len()
        )));
    }
    match (outputs.order_logits_shape, outputs.order_logits) {
        (Some([shape_batch, NUM_QUERIES, NUM_QUERIES]), Some(values))
            if shape_batch == batch_size
                && values.len() == batch_size * NUM_QUERIES * NUM_QUERIES =>
        {
            Ok(())
        }
        (None, None) => Ok(()),
        (shape, values) => Err(LayoutError::InvalidModelOutput(format!(
            "expected order_logits [{batch_size}, {NUM_QUERIES}, {NUM_QUERIES}], got shape {shape:?} and {} values",
            values.map_or(0, <[f32]>::len)
        ))),
    }
}

/// Converts pairwise order logits into a stable per-query ordering rank.
fn compute_order_sequence(order_logits: &[f32]) -> Vec<usize> {
    let mut order_votes = vec![0.0; NUM_QUERIES];
    for col in 0..NUM_QUERIES {
        let mut votes = 0.0;
        for row in 0..NUM_QUERIES {
            if row < col {
                votes += sigmoid(order_logits[row * NUM_QUERIES + col]);
            } else if row > col {
                votes += 1.0 - sigmoid(order_logits[col * NUM_QUERIES + row]);
            }
        }
        order_votes[col] = votes;
    }

    let mut order_pointers: Vec<usize> = (0..NUM_QUERIES).collect();
    order_pointers.sort_by(|a, b| order_votes[*a].total_cmp(&order_votes[*b]));
    let mut order_seq = vec![0; NUM_QUERIES];
    for (rank, pointer) in order_pointers.into_iter().enumerate() {
        order_seq[pointer] = rank;
    }
    order_seq
}

/// Applies the scalar sigmoid used for class scores and order logits.
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}
