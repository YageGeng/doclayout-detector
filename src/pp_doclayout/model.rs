use super::backbone::HgNetV2Backbone;
use super::encoder::{PPDocLayoutV3EncoderInputProjection, PPDocLayoutV3HybridEncoder};
use super::layers::{Activation, ConvNormAct, load_layer_norm, load_linear};
use super::weights::PPDocLayoutV3Weights;
use crate::error::LayoutError;
use burn::tensor::activation::{relu, sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::ops::{GridSampleOptions, InterpolateMode};
use burn::tensor::{Int, Tensor, TensorData};
use burn_nn::{LayerNorm, Linear};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct PPDocLayoutV3Model<B: Backend> {
    backbone: HgNetV2Backbone<B>,
    encoder_input_proj: PPDocLayoutV3EncoderInputProjection<B>,
    encoder: PPDocLayoutV3HybridEncoder<B>,
    decoder_input_proj: PPDocLayoutV3DecoderInputProjection<B>,
    enc_output_linear: Linear<B>,
    enc_output_norm: LayerNorm<B>,
    enc_score_head: Linear<B>,
    enc_bbox_head: PPDocLayoutV3MlpPredictionHead<B>,
    decoder: PPDocLayoutV3Decoder<B>,
    decoder_norm: LayerNorm<B>,
    decoder_order_head: Vec<Linear<B>>,
    decoder_global_pointer: PPDocLayoutV3GlobalPointer<B>,
    mask_query_head: PPDocLayoutV3MlpPredictionHead<B>,
}

#[derive(Debug, Clone)]
pub struct PPDocLayoutV3ProposalOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub pred_boxes: Tensor<B, 3>,
}

#[derive(Debug, Clone)]
pub struct PPDocLayoutV3Output<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub pred_boxes: Tensor<B, 3>,
    pub order_logits: Option<Tensor<B, 3>>,
}

impl<B: Backend> PPDocLayoutV3Model<B> {
    pub fn load(path: &Path, device: &B::Device) -> Result<Self, LayoutError> {
        let weights = PPDocLayoutV3Weights::from_file(path)?;
        Self::load_weights(&weights, device)
    }

    pub fn load_weights(
        weights: &PPDocLayoutV3Weights,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Ok(Self {
            backbone: HgNetV2Backbone::load(weights, "model.backbone.model", device)?,
            encoder_input_proj: PPDocLayoutV3EncoderInputProjection::load(
                weights,
                "model.encoder_input_proj",
                device,
            )?,
            encoder: PPDocLayoutV3HybridEncoder::load(weights, "model.encoder", device)?,
            decoder_input_proj: PPDocLayoutV3DecoderInputProjection::load(
                weights,
                "model.decoder_input_proj",
                device,
            )?,
            enc_output_linear: load_linear(weights, "model.enc_output.0", 256, 256, true, device)?,
            enc_output_norm: load_layer_norm(weights, "model.enc_output.1", 256, device)?,
            enc_score_head: load_linear(weights, "model.enc_score_head", 256, 25, true, device)?,
            enc_bbox_head: PPDocLayoutV3MlpPredictionHead::load(
                weights,
                "model.enc_bbox_head",
                256,
                256,
                4,
                3,
                device,
            )?,
            decoder: PPDocLayoutV3Decoder::load(weights, "model.decoder", device)?,
            decoder_norm: load_layer_norm(weights, "model.decoder_norm", 256, device)?,
            decoder_order_head: (0..6)
                .map(|index| {
                    load_linear(
                        weights,
                        &format!("model.decoder_order_head.{index}"),
                        256,
                        256,
                        true,
                        device,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            decoder_global_pointer: PPDocLayoutV3GlobalPointer::load(
                weights,
                "model.decoder_global_pointer",
                device,
            )?,
            mask_query_head: PPDocLayoutV3MlpPredictionHead::load(
                weights,
                "model.mask_query_head",
                256,
                256,
                32,
                3,
                device,
            )?,
        })
    }

    pub fn forward(&self, pixel_values: Tensor<B, 4>) -> PPDocLayoutV3Output<B> {
        let prepared = self.prepare_decoder_inputs(pixel_values);
        self.forward_prepared(prepared)
    }

    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    pub async fn forward_async(
        &self,
        pixel_values: Tensor<B, 4>,
    ) -> Result<PPDocLayoutV3Output<B>, LayoutError> {
        let prepared = self.prepare_decoder_inputs_async(pixel_values).await?;
        Ok(self.forward_prepared(prepared))
    }

    fn forward_prepared(&self, prepared: PPDocLayoutV3PreparedInputs<B>) -> PPDocLayoutV3Output<B> {
        let decoder_output = self.decoder.forward(
            prepared.target,
            prepared.reference_points_unact,
            prepared.source_flatten,
            prepared.spatial_shapes,
        );
        let out_query = self.decoder_norm.forward(decoder_output.hidden_states);
        let logits = self.enc_score_head.forward(out_query.clone());
        let order_query = self.decoder_order_head[5].forward(out_query);
        let order_logits = self.decoder_global_pointer.forward(order_query);

        PPDocLayoutV3Output {
            logits,
            pred_boxes: decoder_output.reference_points,
            order_logits: Some(order_logits),
        }
    }

    pub fn forward_encoder_proposals(
        &self,
        pixel_values: Tensor<B, 4>,
    ) -> PPDocLayoutV3ProposalOutput<B> {
        let prepared = self.prepare_decoder_inputs(pixel_values);

        PPDocLayoutV3ProposalOutput {
            logits: prepared.encoder_logits,
            pred_boxes: sigmoid(prepared.encoder_reference_points_unact),
        }
    }

    fn prepare_decoder_inputs(&self, pixel_values: Tensor<B, 4>) -> PPDocLayoutV3PreparedInputs<B> {
        let encoded = self.encode_decoder_inputs(pixel_values);
        let topk_indices = proposal_topk_indices(encoded.proposal_scores.clone(), encoded.topk);
        self.prepare_decoder_inputs_from_topk(encoded, topk_indices)
    }

    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    async fn prepare_decoder_inputs_async(
        &self,
        pixel_values: Tensor<B, 4>,
    ) -> Result<PPDocLayoutV3PreparedInputs<B>, LayoutError> {
        let encoded = self.encode_decoder_inputs(pixel_values);
        let topk_indices =
            proposal_topk_indices_async(encoded.proposal_scores.clone(), encoded.topk)
                .await
                .map_err(|error| LayoutError::InvalidModelOutput(error.to_string()))?;
        Ok(self.prepare_decoder_inputs_from_topk(encoded, topk_indices))
    }

    fn encode_decoder_inputs(&self, pixel_values: Tensor<B, 4>) -> PPDocLayoutV3EncodedInputs<B> {
        debug_stats("pixel", &pixel_values);
        let features = self.backbone.forward(pixel_values);
        for (index, feature) in features.iter().enumerate() {
            debug_stats(&format!("backbone {index}"), feature);
        }
        let projected = self.encoder_input_proj.forward(vec![
            features[1].clone(),
            features[2].clone(),
            features[3].clone(),
        ]);
        for (index, feature) in projected.iter().enumerate() {
            debug_stats(&format!("proj {index}"), feature);
        }
        let encoder_outputs = self.encoder.forward(projected, vec![features[0].clone()]);
        for (index, feature) in encoder_outputs.last_hidden_state.iter().enumerate() {
            debug_stats(&format!("enc {index}"), feature);
        }
        debug_stats("mask_feat", &encoder_outputs.mask_feat);
        let mask_feat = encoder_outputs.mask_feat.clone();
        let sources = self
            .decoder_input_proj
            .forward(encoder_outputs.last_hidden_state);
        for (index, source) in sources.iter().enumerate() {
            debug_stats(&format!("source {index}"), source);
        }

        let mut source_flatten = Vec::with_capacity(sources.len());
        let mut spatial_shapes = Vec::with_capacity(sources.len());
        for source in sources {
            let [_, _, height, width] = source.dims();
            spatial_shapes.push((height, width));
            source_flatten.push(source.flatten(2, 3).swap_dims(1, 2));
        }
        let source_flatten = Tensor::cat(source_flatten, 1);
        let (anchors, valid_mask) =
            anchors_and_valid_mask(&spatial_shapes, source_flatten.device());
        let memory = source_flatten.clone() * valid_mask.clone().repeat_dim(2, 256);

        let output_memory = self
            .enc_output_norm
            .forward(self.enc_output_linear.forward(memory));
        debug_stats("output_memory", &output_memory);
        let enc_outputs_class = self.enc_score_head.forward(output_memory.clone());
        debug_stats("enc_outputs_class", &enc_outputs_class);
        let enc_outputs_coord_logits = self.enc_bbox_head.forward(output_memory.clone()) + anchors;
        let total_tokens = enc_outputs_class.dims()[1];
        let topk = total_tokens.min(300);
        let proposal_scores = enc_outputs_class.clone().max_dim(2)
            - valid_mask.clone().neg().add_scalar(1.0).mul_scalar(1.0e8);

        PPDocLayoutV3EncodedInputs {
            output_memory,
            enc_outputs_class,
            enc_outputs_coord_logits,
            mask_feat,
            source_flatten,
            spatial_shapes,
            proposal_scores,
            topk,
        }
    }

    fn prepare_decoder_inputs_from_topk(
        &self,
        encoded: PPDocLayoutV3EncodedInputs<B>,
        topk_indices: Tensor<B, 3, Int>,
    ) -> PPDocLayoutV3PreparedInputs<B> {
        let encoder_logits = pad_queries(
            encoded
                .enc_outputs_class
                .gather(1, topk_indices.clone().repeat_dim(2, 25)),
            25,
        );
        let encoder_reference_points_unact = pad_queries(
            encoded
                .enc_outputs_coord_logits
                .gather(1, topk_indices.clone().repeat_dim(2, 4)),
            4,
        );
        let target = pad_queries(
            encoded
                .output_memory
                .gather(1, topk_indices.repeat_dim(2, 256)),
            256,
        );
        let out_query = self.decoder_norm.forward(target.clone());
        let mask_query_embed = self.mask_query_head.forward(out_query);
        let [batch_size, query_count, _] = mask_query_embed.dims();
        let [_, _, mask_height, mask_width] = encoded.mask_feat.dims();
        let enc_out_masks = mask_query_embed
            .matmul(encoded.mask_feat.flatten(2, 3))
            .reshape([batch_size, query_count, mask_height, mask_width]);
        let reference_points_unact = inverse_sigmoid(mask_to_box_coordinate(enc_out_masks));

        PPDocLayoutV3PreparedInputs {
            target,
            reference_points_unact,
            encoder_reference_points_unact,
            source_flatten: encoded.source_flatten,
            spatial_shapes: encoded.spatial_shapes,
            encoder_logits,
        }
    }
}

struct PPDocLayoutV3EncodedInputs<B: Backend> {
    output_memory: Tensor<B, 3>,
    enc_outputs_class: Tensor<B, 3>,
    enc_outputs_coord_logits: Tensor<B, 3>,
    mask_feat: Tensor<B, 4>,
    source_flatten: Tensor<B, 3>,
    spatial_shapes: Vec<(usize, usize)>,
    proposal_scores: Tensor<B, 3>,
    topk: usize,
}

struct PPDocLayoutV3PreparedInputs<B: Backend> {
    target: Tensor<B, 3>,
    reference_points_unact: Tensor<B, 3>,
    encoder_reference_points_unact: Tensor<B, 3>,
    source_flatten: Tensor<B, 3>,
    spatial_shapes: Vec<(usize, usize)>,
    encoder_logits: Tensor<B, 3>,
}

fn pad_queries<B: Backend>(tensor: Tensor<B, 3>, channels: usize) -> Tensor<B, 3> {
    let query_count = tensor.dims()[1];
    if query_count >= 300 {
        return tensor;
    }
    let repeat = 300_usize.div_ceil(query_count);
    tensor
        .repeat_dim(1, repeat)
        .slice([0..1, 0..300, 0..channels])
}

fn proposal_topk_indices<B: Backend>(scores: Tensor<B, 3>, topk: usize) -> Tensor<B, 3, Int> {
    let (_, indices) = scores.topk_with_indices(topk, 1);
    indices
}

#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
async fn proposal_topk_indices_async<B: Backend>(
    scores: Tensor<B, 3>,
    topk: usize,
) -> Result<Tensor<B, 3, Int>, String> {
    let device = scores.device();
    let dims = scores.dims();
    let values = scores
        .into_data_async()
        .await
        .map_err(|error| format!("read proposal scores tensor: {error}"))?
        .to_vec::<f32>()
        .map_err(|error| format!("decode proposal scores tensor: {error}"))?;
    Ok(host_topk_indices_from_values(values, dims, topk, &device))
}

#[cfg(test)]
fn host_topk_indices<B: Backend>(scores: Tensor<B, 3>, topk: usize) -> Tensor<B, 3, Int> {
    let device = scores.device();
    let dims = scores.dims();
    let values = scores
        .into_data()
        .to_vec::<f32>()
        .expect("failed to read proposal scores for host top-k");
    host_topk_indices_from_values(values, dims, topk, &device)
}

#[cfg(any(test, all(target_family = "wasm", feature = "backend-webgpu")))]
fn host_topk_indices_from_values<B: Backend>(
    values: Vec<f32>,
    dims: [usize; 3],
    topk: usize,
    device: &B::Device,
) -> Tensor<B, 3, Int> {
    let [batch_size, token_count, channels] = dims;
    assert_eq!(
        channels, 1,
        "proposal scores are expected to have a singleton channel dimension"
    );
    let mut indices = Vec::with_capacity(batch_size * topk);
    for batch_index in 0..batch_size {
        let batch_offset = batch_index * token_count;
        let mut ranked = (0..token_count).collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| {
            values[batch_offset + *right].total_cmp(&values[batch_offset + *left])
        });
        indices.extend(ranked.into_iter().take(topk).map(|index| index as i64));
    }

    Tensor::<B, 2, Int>::from_data(TensorData::new(indices, [batch_size, topk]), device)
        .reshape([batch_size, topk, 1])
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3DecoderInputProjection<B: Backend> {
    projections: Vec<ConvNormAct<B>>,
}

impl<B: Backend> PPDocLayoutV3DecoderInputProjection<B> {
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        let mut projections = Vec::with_capacity(3);
        for index in 0..3 {
            projections.push(ConvNormAct::load(
                weights,
                &format!("{prefix}.{index}"),
                "0",
                "1",
                256,
                256,
                1,
                1,
                Activation::None,
                device,
            )?);
        }
        Ok(Self { projections })
    }

    fn forward(&self, inputs: Vec<Tensor<B, 4>>) -> Vec<Tensor<B, 4>> {
        self.projections
            .iter()
            .zip(inputs)
            .map(|(projection, input)| projection.forward(input))
            .collect()
    }
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3MlpPredictionHead<B: Backend> {
    layers: Vec<Linear<B>>,
}

impl<B: Backend> PPDocLayoutV3MlpPredictionHead<B> {
    #[allow(clippy::too_many_arguments)]
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        let mut layers = Vec::with_capacity(num_layers);
        for index in 0..num_layers {
            let d_input = if index == 0 { input_dim } else { hidden_dim };
            let d_output = if index + 1 == num_layers {
                output_dim
            } else {
                hidden_dim
            };
            layers.push(load_linear(
                weights,
                &format!("{prefix}.layers.{index}"),
                d_input,
                d_output,
                true,
                device,
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, mut input: Tensor<B, 3>) -> Tensor<B, 3> {
        for (index, layer) in self.layers.iter().enumerate() {
            input = layer.forward(input);
            if index + 1 != self.layers.len() {
                input = relu(input);
            }
        }
        input
    }
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3DecoderOutput<B: Backend> {
    hidden_states: Tensor<B, 3>,
    reference_points: Tensor<B, 3>,
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3Decoder<B: Backend> {
    layers: Vec<PPDocLayoutV3DecoderLayer<B>>,
    query_pos_head: PPDocLayoutV3MlpPredictionHead<B>,
    bbox_embed: PPDocLayoutV3MlpPredictionHead<B>,
}

impl<B: Backend> PPDocLayoutV3Decoder<B> {
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        let layers = (0..6)
            .map(|index| {
                PPDocLayoutV3DecoderLayer::load(
                    weights,
                    &format!("{prefix}.layers.{index}"),
                    device,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layers,
            query_pos_head: PPDocLayoutV3MlpPredictionHead::load(
                weights,
                &format!("{prefix}.query_pos_head"),
                4,
                512,
                256,
                2,
                device,
            )?,
            // Tied in the HF model: decoder.bbox_embed uses model.enc_bbox_head.
            bbox_embed: PPDocLayoutV3MlpPredictionHead::load(
                weights,
                "model.enc_bbox_head",
                256,
                256,
                4,
                3,
                device,
            )?,
        })
    }

    fn forward(
        &self,
        mut hidden_states: Tensor<B, 3>,
        reference_points_unact: Tensor<B, 3>,
        encoder_hidden_states: Tensor<B, 3>,
        spatial_shapes: Vec<(usize, usize)>,
    ) -> PPDocLayoutV3DecoderOutput<B> {
        let mut reference_points = sigmoid(reference_points_unact);
        for layer in &self.layers {
            let query_pos = self.query_pos_head.forward(reference_points.clone());
            hidden_states = layer.forward(
                hidden_states,
                query_pos,
                reference_points.clone(),
                encoder_hidden_states.clone(),
                &spatial_shapes,
            );
            reference_points = sigmoid(
                self.bbox_embed.forward(hidden_states.clone()) + inverse_sigmoid(reference_points),
            );
        }

        PPDocLayoutV3DecoderOutput {
            hidden_states,
            reference_points,
        }
    }
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3DecoderLayer<B: Backend> {
    self_attn: PPDocLayoutV3SelfAttention<B>,
    self_attn_layer_norm: LayerNorm<B>,
    encoder_attn: PPDocLayoutV3MultiscaleDeformableAttention<B>,
    encoder_attn_layer_norm: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    final_layer_norm: LayerNorm<B>,
}

impl<B: Backend> PPDocLayoutV3DecoderLayer<B> {
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Ok(Self {
            self_attn: PPDocLayoutV3SelfAttention::load(
                weights,
                &format!("{prefix}.self_attn"),
                device,
            )?,
            self_attn_layer_norm: load_layer_norm(
                weights,
                &format!("{prefix}.self_attn_layer_norm"),
                256,
                device,
            )?,
            encoder_attn: PPDocLayoutV3MultiscaleDeformableAttention::load(
                weights,
                &format!("{prefix}.encoder_attn"),
                device,
            )?,
            encoder_attn_layer_norm: load_layer_norm(
                weights,
                &format!("{prefix}.encoder_attn_layer_norm"),
                256,
                device,
            )?,
            fc1: load_linear(weights, &format!("{prefix}.fc1"), 256, 1024, true, device)?,
            fc2: load_linear(weights, &format!("{prefix}.fc2"), 1024, 256, true, device)?,
            final_layer_norm: load_layer_norm(
                weights,
                &format!("{prefix}.final_layer_norm"),
                256,
                device,
            )?,
        })
    }

    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        query_pos: Tensor<B, 3>,
        reference_points: Tensor<B, 3>,
        encoder_hidden_states: Tensor<B, 3>,
        spatial_shapes: &[(usize, usize)],
    ) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = self
            .self_attn
            .forward(hidden_states, Some(query_pos.clone()));
        let hidden_states = self.self_attn_layer_norm.forward(residual + hidden_states);

        let residual = hidden_states.clone();
        let hidden_states = self.encoder_attn.forward(
            hidden_states,
            query_pos,
            reference_points,
            encoder_hidden_states,
            spatial_shapes,
        );
        let hidden_states = self
            .encoder_attn_layer_norm
            .forward(residual + hidden_states);

        let residual = hidden_states.clone();
        let hidden_states = self.fc2.forward(relu(self.fc1.forward(hidden_states)));
        self.final_layer_norm.forward(residual + hidden_states)
    }
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3SelfAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
}

impl<B: Backend> PPDocLayoutV3SelfAttention<B> {
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Ok(Self {
            q_proj: load_linear(weights, &format!("{prefix}.q_proj"), 256, 256, true, device)?,
            k_proj: load_linear(weights, &format!("{prefix}.k_proj"), 256, 256, true, device)?,
            v_proj: load_linear(weights, &format!("{prefix}.v_proj"), 256, 256, true, device)?,
            out_proj: load_linear(
                weights,
                &format!("{prefix}.out_proj"),
                256,
                256,
                true,
                device,
            )?,
        })
    }

    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        position_embeddings: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = hidden_states.dims();
        let query_key_input = match position_embeddings {
            Some(position_embeddings) => hidden_states.clone() + position_embeddings,
            None => hidden_states.clone(),
        };
        let query = self
            .q_proj
            .forward(query_key_input.clone())
            .reshape([batch_size, seq_len, 8, 32])
            .swap_dims(1, 2);
        let key = self
            .k_proj
            .forward(query_key_input)
            .reshape([batch_size, seq_len, 8, 32])
            .swap_dims(1, 2);
        let value = self
            .v_proj
            .forward(hidden_states)
            .reshape([batch_size, seq_len, 8, 32])
            .swap_dims(1, 2);
        let weights = softmax(query.matmul(key.transpose()).div_scalar(32.0_f32.sqrt()), 3);
        let context = weights
            .matmul(value)
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, 256]);
        self.out_proj.forward(context)
    }
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3MultiscaleDeformableAttention<B: Backend> {
    sampling_offsets: Linear<B>,
    attention_weights: Linear<B>,
    value_proj: Linear<B>,
    output_proj: Linear<B>,
}

impl<B: Backend> PPDocLayoutV3MultiscaleDeformableAttention<B> {
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Ok(Self {
            sampling_offsets: load_linear(
                weights,
                &format!("{prefix}.sampling_offsets"),
                256,
                192,
                true,
                device,
            )?,
            attention_weights: load_linear(
                weights,
                &format!("{prefix}.attention_weights"),
                256,
                96,
                true,
                device,
            )?,
            value_proj: load_linear(
                weights,
                &format!("{prefix}.value_proj"),
                256,
                256,
                true,
                device,
            )?,
            output_proj: load_linear(
                weights,
                &format!("{prefix}.output_proj"),
                256,
                256,
                true,
                device,
            )?,
        })
    }

    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        position_embeddings: Tensor<B, 3>,
        reference_points: Tensor<B, 3>,
        encoder_hidden_states: Tensor<B, 3>,
        spatial_shapes: &[(usize, usize)],
    ) -> Tensor<B, 3> {
        let [batch_size, num_queries, _] = hidden_states.dims();
        let sequence_length = encoder_hidden_states.dims()[1];
        let hidden_states = hidden_states + position_embeddings;
        let value = self.value_proj.forward(encoder_hidden_states).reshape([
            batch_size,
            sequence_length,
            8,
            32,
        ]);
        let offsets = self
            .sampling_offsets
            .forward(hidden_states.clone())
            .reshape([batch_size, num_queries, 8, 3, 4, 2]);
        let attn = softmax(
            self.attention_weights
                .forward(hidden_states)
                .reshape([batch_size, num_queries, 8, 12]),
            3,
        )
        .reshape([batch_size, num_queries, 8, 3, 4]);

        let ref_xy = reference_points
            .clone()
            .slice([0..batch_size, 0..num_queries, 0..2])
            .reshape([batch_size, num_queries, 1, 1, 1, 2]);
        let ref_wh = reference_points
            .slice([0..batch_size, 0..num_queries, 2..4])
            .reshape([batch_size, num_queries, 1, 1, 1, 2]);
        let sampling_locations = ref_xy + offsets.div_scalar(4.0) * ref_wh.mul_scalar(0.5);
        let sampling_grids = sampling_locations.mul_scalar(2.0).sub_scalar(1.0);

        let mut sampled_levels = Vec::with_capacity(spatial_shapes.len());
        let mut start = 0usize;
        for (level, (height, width)) in spatial_shapes.iter().copied().enumerate() {
            let len = height * width;
            let value_l: Tensor<B, 4> =
                value
                    .clone()
                    .slice([0..batch_size, start..start + len, 0..8, 0..32]);
            let value_l = value_l
                .reshape([batch_size, len, 256])
                .swap_dims(1, 2)
                .reshape([batch_size * 8, 32, height, width]);
            let grid_l = sampling_grids
                .clone()
                .slice([
                    0..batch_size,
                    0..num_queries,
                    0..8,
                    level..level + 1,
                    0..4,
                    0..2,
                ])
                .reshape([batch_size, num_queries, 8, 4, 2])
                .swap_dims(1, 2)
                .flatten(0, 1);
            sampled_levels.push(
                value_l.grid_sample_2d(grid_l, GridSampleOptions::new(InterpolateMode::Bilinear)),
            );
            start += len;
        }
        let sampled = Tensor::cat(sampled_levels, 3);
        let attn = attn
            .swap_dims(1, 2)
            .reshape([batch_size * 8, 1, num_queries, 12]);
        let output = (sampled * attn)
            .sum_dim(3)
            .reshape([batch_size, 256, num_queries])
            .swap_dims(1, 2)
            .reshape([batch_size, num_queries, 256]);
        self.output_proj.forward(output)
    }
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3GlobalPointer<B: Backend> {
    dense: Linear<B>,
}

impl<B: Backend> PPDocLayoutV3GlobalPointer<B> {
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Ok(Self {
            dense: load_linear(weights, &format!("{prefix}.dense"), 256, 128, true, device)?,
        })
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input.dims();
        let projected = self
            .dense
            .forward(input)
            .reshape([batch_size, seq_len, 2, 64]);
        let queries = projected
            .clone()
            .slice([0..batch_size, 0..seq_len, 0..1, 0..64])
            .reshape([batch_size, seq_len, 64]);
        let keys = projected
            .slice([0..batch_size, 0..seq_len, 1..2, 0..64])
            .reshape([batch_size, seq_len, 64]);
        let logits = queries
            .matmul(keys.swap_dims(1, 2))
            .div_scalar(64.0_f32.sqrt());
        mask_order_logits(logits)
    }
}

fn mask_order_logits<B: Backend>(logits: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch_size, seq_len, _] = logits.dims();
    let device = logits.device();
    let mut mask = Vec::with_capacity(seq_len * seq_len);
    for row in 0..seq_len {
        for col in 0..seq_len {
            mask.push(row >= col);
        }
    }
    let mask = Tensor::from_data(TensorData::new(mask, [1, seq_len, seq_len]), &device)
        .repeat_dim(0, batch_size);
    logits.mask_fill(mask, -1.0e4)
}

fn inverse_sigmoid<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
    let input = input.clamp(0.0, 1.0);
    let x1 = input.clone().clamp_min(1e-5);
    let x2 = input.neg().add_scalar(1.0).clamp_min(1e-5);
    (x1 / x2).log()
}

fn anchors_and_valid_mask<B: Backend>(
    spatial_shapes: &[(usize, usize)],
    device: B::Device,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let mut anchors = Vec::new();
    let mut valid_mask = Vec::new();
    for (level, (height, width)) in spatial_shapes.iter().copied().enumerate() {
        let wh = 0.05_f32 * 2.0_f32.powi(level as i32);
        for y in 0..height {
            for x in 0..width {
                let cx = (x as f32 + 0.5) / width as f32;
                let cy = (y as f32 + 0.5) / height as f32;
                let valid = cx > 0.01 && cx < 0.99 && cy > 0.01 && cy < 0.99 && wh < 0.99;
                valid_mask.push(if valid { 1.0 } else { 0.0 });
                if valid {
                    anchors.push(logit(cx));
                    anchors.push(logit(cy));
                    anchors.push(logit(wh));
                    anchors.push(logit(wh));
                } else {
                    anchors.extend([1.0e8; 4]);
                }
            }
        }
    }
    let total = valid_mask.len();
    (
        Tensor::from_data(TensorData::new(anchors, [1, total, 4]), &device),
        Tensor::from_data(TensorData::new(valid_mask, [1, total, 1]), &device),
    )
}

fn logit(value: f32) -> f32 {
    (value / (1.0 - value)).ln()
}

fn mask_to_box_coordinate<B: Backend>(masks: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch_size, query_count, height, width] = masks.dims();
    let device = masks.device();
    let mask = masks.greater_elem(0.0);
    let zeros = Tensor::<B, 4>::zeros([batch_size, query_count, height, width], &device);

    let mut x_values = Vec::with_capacity(height * width);
    let mut y_values = Vec::with_capacity(height * width);
    for y in 0..height {
        for x in 0..width {
            x_values.push(x as f32);
            y_values.push(y as f32);
        }
    }
    let x_coords =
        Tensor::<B, 4>::from_data(TensorData::new(x_values, [1, 1, height, width]), &device)
            .repeat_dim(0, batch_size)
            .repeat_dim(1, query_count);
    let y_coords =
        Tensor::<B, 4>::from_data(TensorData::new(y_values, [1, 1, height, width]), &device)
            .repeat_dim(0, batch_size)
            .repeat_dim(1, query_count);

    let mask_values = zeros.clone().mask_fill(mask.clone(), 1.0);
    let non_empty = mask_values.flatten(2, 3).sum_dim(2).greater_elem(0.0);

    let x_masked = zeros.clone().mask_where(mask.clone(), x_coords.clone());
    let y_masked = zeros.clone().mask_where(mask.clone(), y_coords.clone());
    let x_max = x_masked.flatten(2, 3).max_dim(2).add_scalar(1.0);
    let y_max = y_masked.flatten(2, 3).max_dim(2).add_scalar(1.0);

    let full = Tensor::<B, 4>::full([batch_size, query_count, height, width], f32::MAX, &device);
    let x_min = full
        .clone()
        .mask_where(mask.clone(), x_coords)
        .flatten(2, 3)
        .min_dim(2);
    let y_min = full.mask_where(mask, y_coords).flatten(2, 3).min_dim(2);

    let x_min = x_min.div_scalar(width as f32);
    let x_max = x_max.div_scalar(width as f32);
    let y_min = y_min.div_scalar(height as f32);
    let y_max = y_max.div_scalar(height as f32);

    let center_x = (x_min.clone() + x_max.clone()).div_scalar(2.0);
    let center_y = (y_min.clone() + y_max.clone()).div_scalar(2.0);
    let box_width = x_max - x_min;
    let box_height = y_max - y_min;
    Tensor::cat(vec![center_x, center_y, box_width, box_height], 2)
        .mask_fill(non_empty.bool_not().repeat_dim(2, 4), 0.0)
}

fn debug_stats<B: Backend, const D: usize>(name: &str, tensor: &Tensor<B, D>) {
    if std::env::var_os("LITEPARSE_LAYOUT_DEBUG_STATS").is_none() {
        return;
    }
    let dims = tensor.dims();
    let Ok(values) = tensor.clone().into_data().to_vec::<f32>() else {
        tracing::debug!(name, ?dims, "failed to read tensor data");
        return;
    };
    if values.is_empty() {
        tracing::debug!(name, ?dims, "tensor data is empty");
        return;
    }
    let len = values.len() as f32;
    let mean = values.iter().sum::<f32>() / len;
    let var = values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / len.max(1.0);
    let max_abs = values
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    tracing::debug!(name, ?dims, mean, std = var.sqrt(), max_abs, "tensor stats");
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn mask_to_box_coordinate_returns_normalized_cxcywh_and_zero_for_empty_masks() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let mut values = vec![-1.0; 2 * 4 * 5];
        for y in 1..=2 {
            for x in 2..=3 {
                values[((y * 5) + x) as usize] = 1.0;
            }
        }
        let masks =
            Tensor::<TestBackend, 4>::from_data(TensorData::new(values, [1, 2, 4, 5]), &device);

        let boxes = mask_to_box_coordinate(masks);
        let values = boxes.into_data().to_vec::<f32>().unwrap();

        assert!((values[0] - 0.6).abs() < 1e-6);
        assert!((values[1] - 0.5).abs() < 1e-6);
        assert!((values[2] - 0.4).abs() < 1e-6);
        assert!((values[3] - 0.5).abs() < 1e-6);
        assert_eq!(&values[4..8], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn host_topk_indices_selects_highest_proposal_scores() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let scores = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.1, 0.7, -2.0, 1.3, 0.5], [1, 5, 1]),
            &device,
        );

        let indices = host_topk_indices(scores, 3);
        let values = indices.into_data().to_vec::<i64>().unwrap();

        assert_eq!(values, vec![3, 1, 4]);
    }

    #[test]
    fn order_logits_mask_sets_lower_triangle_and_diagonal_to_negative_sentinel() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let logits =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(vec![1.0; 9], [1, 3, 3]), &device);

        let masked = mask_order_logits(logits);
        let values = masked.into_data().to_vec::<f32>().unwrap();

        assert_eq!(values[0], -1.0e4);
        assert_eq!(values[1], 1.0);
        assert_eq!(values[2], 1.0);
        assert_eq!(values[3], -1.0e4);
        assert_eq!(values[4], -1.0e4);
        assert_eq!(values[5], 1.0);
        assert_eq!(values[6], -1.0e4);
        assert_eq!(values[7], -1.0e4);
        assert_eq!(values[8], -1.0e4);
    }
}
