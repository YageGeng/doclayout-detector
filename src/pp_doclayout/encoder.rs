use super::layers::{Activation, ConvBnAct, ConvNormAct, load_layer_norm, load_linear};
use super::weights::PPDocLayoutV3Weights;
use crate::error::LayoutError;
use burn::module::Param;
use burn::tensor::activation::{gelu, silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Tensor, TensorData};
use burn_nn::conv::{Conv2d, Conv2dConfig};
use burn_nn::{LayerNorm, Linear};

#[derive(Debug, Clone)]
pub struct PPDocLayoutV3EncoderInputProjection<B: Backend> {
    projections: Vec<ConvNormAct<B>>,
}

impl<B: Backend> PPDocLayoutV3EncoderInputProjection<B> {
    /// Loads the 1x1 projections that normalize backbone feature channels to the encoder width.
    pub fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        let channels = [512, 1024, 2048];
        let mut projections = Vec::with_capacity(channels.len());
        for (index, in_channels) in channels.into_iter().enumerate() {
            projections.push(ConvNormAct::load(
                weights,
                &format!("{prefix}.{index}"),
                "0",
                "1",
                in_channels,
                256,
                1,
                1,
                Activation::None,
                device,
            )?);
        }
        Ok(Self { projections })
    }

    /// Projects selected backbone feature maps into the 256-channel encoder space.
    pub fn forward(&self, inputs: Vec<Tensor<B, 4>>) -> Vec<Tensor<B, 4>> {
        self.projections
            .iter()
            .zip(inputs)
            .map(|(projection, input)| projection.forward(input))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct PPDocLayoutV3HybridEncoderOutput<B: Backend> {
    pub last_hidden_state: Vec<Tensor<B, 4>>,
    pub mask_feat: Tensor<B, 4>,
}

#[derive(Debug, Clone)]
pub struct PPDocLayoutV3HybridEncoder<B: Backend> {
    aifi: PPDocLayoutV3AifiLayer<B>,
    lateral_convs: Vec<ConvNormAct<B>>,
    fpn_blocks: Vec<PPDocLayoutV3CspRepLayer<B>>,
    downsample_convs: Vec<ConvNormAct<B>>,
    pan_blocks: Vec<PPDocLayoutV3CspRepLayer<B>>,
    mask_feature_head: PPDocLayoutV3MaskFeatFpn<B>,
    encoder_mask_lateral: ConvBnAct<B>,
    encoder_mask_output: PPDocLayoutV3EncoderMaskOutput<B>,
}

impl<B: Backend> PPDocLayoutV3HybridEncoder<B> {
    /// Loads the hybrid encoder, FPN/PAN fusion blocks, and mask feature head.
    pub fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Ok(Self {
            aifi: PPDocLayoutV3AifiLayer::load(
                weights,
                &format!("{prefix}.encoder.0.layers.0"),
                device,
            )?,
            lateral_convs: vec![
                ConvNormAct::load(
                    weights,
                    &format!("{prefix}.lateral_convs.0"),
                    "conv",
                    "norm",
                    256,
                    256,
                    1,
                    1,
                    Activation::Silu,
                    device,
                )?,
                ConvNormAct::load(
                    weights,
                    &format!("{prefix}.lateral_convs.1"),
                    "conv",
                    "norm",
                    256,
                    256,
                    1,
                    1,
                    Activation::Silu,
                    device,
                )?,
            ],
            fpn_blocks: vec![
                PPDocLayoutV3CspRepLayer::load(weights, &format!("{prefix}.fpn_blocks.0"), device)?,
                PPDocLayoutV3CspRepLayer::load(weights, &format!("{prefix}.fpn_blocks.1"), device)?,
            ],
            downsample_convs: vec![
                ConvNormAct::load(
                    weights,
                    &format!("{prefix}.downsample_convs.0"),
                    "conv",
                    "norm",
                    256,
                    256,
                    3,
                    2,
                    Activation::Silu,
                    device,
                )?,
                ConvNormAct::load(
                    weights,
                    &format!("{prefix}.downsample_convs.1"),
                    "conv",
                    "norm",
                    256,
                    256,
                    3,
                    2,
                    Activation::Silu,
                    device,
                )?,
            ],
            pan_blocks: vec![
                PPDocLayoutV3CspRepLayer::load(weights, &format!("{prefix}.pan_blocks.0"), device)?,
                PPDocLayoutV3CspRepLayer::load(weights, &format!("{prefix}.pan_blocks.1"), device)?,
            ],
            mask_feature_head: PPDocLayoutV3MaskFeatFpn::load(
                weights,
                &format!("{prefix}.mask_feature_head"),
                device,
            )?,
            encoder_mask_lateral: ConvBnAct::load(
                weights,
                &format!("{prefix}.encoder_mask_lateral"),
                128,
                64,
                3,
                1,
                1,
                Activation::Silu,
                device,
            )?,
            encoder_mask_output: PPDocLayoutV3EncoderMaskOutput::load(
                weights,
                &format!("{prefix}.encoder_mask_output"),
                device,
            )?,
        })
    }

    /// Runs AIFI attention, top-down FPN fusion, bottom-up PAN fusion, and mask feature decoding.
    pub fn forward(
        &self,
        mut feature_maps: Vec<Tensor<B, 4>>,
        x4_feat: Vec<Tensor<B, 4>>,
    ) -> PPDocLayoutV3HybridEncoderOutput<B> {
        feature_maps[2] = self.aifi.forward(feature_maps[2].clone());

        let mut fpn_feature_maps = vec![feature_maps[2].clone()];
        for idx in 0..self.fpn_blocks.len() {
            let backbone_feature_map = feature_maps[self.fpn_blocks.len() - idx - 1].clone();
            let top = fpn_feature_maps.last().unwrap().clone();
            let top = self.lateral_convs[idx].forward(top);
            let last_idx = fpn_feature_maps.len() - 1;
            fpn_feature_maps[last_idx] = top.clone();
            let fused = Tensor::cat(vec![upsample_nearest_2x(top), backbone_feature_map], 1);
            let new_fpn_feature_map = self.fpn_blocks[idx].forward(fused);
            fpn_feature_maps.push(new_fpn_feature_map);
        }
        fpn_feature_maps.reverse();

        let mut pan_feature_maps = vec![fpn_feature_maps[0].clone()];
        for idx in 0..self.pan_blocks.len() {
            let top_pan_feature_map = pan_feature_maps.last().unwrap().clone();
            let fpn_feature_map = fpn_feature_maps[idx + 1].clone();
            let downsampled = self.downsample_convs[idx].forward(top_pan_feature_map);
            let fused = Tensor::cat(vec![downsampled, fpn_feature_map], 1);
            pan_feature_maps.push(self.pan_blocks[idx].forward(fused));
        }

        let mut mask_feat = self.mask_feature_head.forward(&pan_feature_maps);
        mask_feat = upsample_bilinear_2x(mask_feat);
        mask_feat = mask_feat + self.encoder_mask_lateral.forward(x4_feat[0].clone());
        mask_feat = self.encoder_mask_output.forward(mask_feat);

        PPDocLayoutV3HybridEncoderOutput {
            last_hidden_state: pan_feature_maps,
            mask_feat,
        }
    }
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3AifiLayer<B: Backend> {
    encoder_layer: PPDocLayoutV3EncoderLayer<B>,
}

impl<B: Backend> PPDocLayoutV3AifiLayer<B> {
    /// Loads the single AIFI transformer layer used on the lowest-resolution feature map.
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Ok(Self {
            encoder_layer: PPDocLayoutV3EncoderLayer::load(weights, prefix, device)?,
        })
    }

    /// Applies positional self-attention over flattened spatial tokens and restores BCHW layout.
    fn forward(&self, hidden_states: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, channels, height, width] = hidden_states.dims();
        let hidden_states = hidden_states.flatten(2, 3).swap_dims(1, 2);
        let position_embeddings =
            position_embeddings::<B>(height, width, channels, hidden_states.device());
        let hidden_states = self
            .encoder_layer
            .forward(hidden_states, Some(position_embeddings));
        hidden_states
            .swap_dims(1, 2)
            .reshape([batch_size, channels, height, width])
    }
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3EncoderLayer<B: Backend> {
    self_attn: PPDocLayoutV3SelfAttention<B>,
    self_attn_layer_norm: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    final_layer_norm: LayerNorm<B>,
}

impl<B: Backend> PPDocLayoutV3EncoderLayer<B> {
    /// Loads one transformer encoder layer with self-attention and feed-forward weights.
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

    /// Runs residual self-attention followed by a residual feed-forward block.
    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        position_embeddings: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = self.self_attn.forward(hidden_states, position_embeddings);
        let hidden_states = self.self_attn_layer_norm.forward(residual + hidden_states);
        let residual = hidden_states.clone();
        let hidden_states = self.fc2.forward(gelu(self.fc1.forward(hidden_states)));
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
    /// Loads query, key, value, and output projections for multi-head self-attention.
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

    /// Computes scaled dot-product attention over flattened encoder tokens.
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
struct PPDocLayoutV3CspRepLayer<B: Backend> {
    conv1: ConvNormAct<B>,
    conv2: ConvNormAct<B>,
    bottlenecks: Vec<PPDocLayoutV3RepVggBlock<B>>,
}

impl<B: Backend> PPDocLayoutV3CspRepLayer<B> {
    /// Loads a CSP Rep layer used by both FPN and PAN feature fusion.
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Ok(Self {
            conv1: ConvNormAct::load(
                weights,
                &format!("{prefix}.conv1"),
                "conv",
                "norm",
                512,
                256,
                1,
                1,
                Activation::Silu,
                device,
            )?,
            conv2: ConvNormAct::load(
                weights,
                &format!("{prefix}.conv2"),
                "conv",
                "norm",
                512,
                256,
                1,
                1,
                Activation::Silu,
                device,
            )?,
            bottlenecks: vec![
                PPDocLayoutV3RepVggBlock::load(
                    weights,
                    &format!("{prefix}.bottlenecks.0"),
                    device,
                )?,
                PPDocLayoutV3RepVggBlock::load(
                    weights,
                    &format!("{prefix}.bottlenecks.1"),
                    device,
                )?,
                PPDocLayoutV3RepVggBlock::load(
                    weights,
                    &format!("{prefix}.bottlenecks.2"),
                    device,
                )?,
            ],
        })
    }

    /// Runs the bottleneck branch and adds it to the shortcut projection branch.
    fn forward(&self, hidden_state: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut hidden_state_1 = self.conv1.forward(hidden_state.clone());
        for block in &self.bottlenecks {
            hidden_state_1 = block.forward(hidden_state_1);
        }
        let hidden_state_2 = self.conv2.forward(hidden_state);
        hidden_state_1 + hidden_state_2
    }
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3RepVggBlock<B: Backend> {
    conv1: ConvNormAct<B>,
    conv2: ConvNormAct<B>,
}

impl<B: Backend> PPDocLayoutV3RepVggBlock<B> {
    /// Loads the two-branch RepVGG block used inside CSP fusion layers.
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Ok(Self {
            conv1: ConvNormAct::load(
                weights,
                &format!("{prefix}.conv1"),
                "conv",
                "norm",
                256,
                256,
                3,
                1,
                Activation::None,
                device,
            )?,
            conv2: ConvNormAct::load(
                weights,
                &format!("{prefix}.conv2"),
                "conv",
                "norm",
                256,
                256,
                1,
                1,
                Activation::None,
                device,
            )?,
        })
    }

    /// Adds the 3x3 and 1x1 convolution branches and applies SiLU activation.
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        silu(self.conv1.forward(input.clone()) + self.conv2.forward(input))
    }
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3MaskFeatFpn<B: Backend> {
    scale_heads: Vec<PPDocLayoutV3ScaleHead<B>>,
    output_conv: ConvBnAct<B>,
}

impl<B: Backend> PPDocLayoutV3MaskFeatFpn<B> {
    /// Loads the multi-scale mask feature FPN heads and output convolution.
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Ok(Self {
            scale_heads: vec![
                PPDocLayoutV3ScaleHead::load(
                    weights,
                    &format!("{prefix}.scale_heads.0"),
                    256,
                    0,
                    device,
                )?,
                PPDocLayoutV3ScaleHead::load(
                    weights,
                    &format!("{prefix}.scale_heads.1"),
                    256,
                    1,
                    device,
                )?,
                PPDocLayoutV3ScaleHead::load(
                    weights,
                    &format!("{prefix}.scale_heads.2"),
                    256,
                    2,
                    device,
                )?,
            ],
            output_conv: ConvBnAct::load(
                weights,
                &format!("{prefix}.output_conv"),
                64,
                64,
                3,
                1,
                1,
                Activation::Silu,
                device,
            )?,
        })
    }

    /// Aligns all PAN features to the highest mask resolution and sums them.
    fn forward(&self, inputs: &[Tensor<B, 4>]) -> Tensor<B, 4> {
        let mut output = self.scale_heads[0].forward(inputs[0].clone());
        let [_, _, height, width] = output.dims();
        for (index, scale_head) in self.scale_heads.iter().enumerate().skip(1) {
            output = output
                + interpolate(
                    scale_head.forward(inputs[index].clone()),
                    [height, width],
                    InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
                );
        }
        self.output_conv.forward(output)
    }
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3ScaleHead<B: Backend> {
    conv0: ConvBnAct<B>,
    conv1: Option<ConvBnAct<B>>,
    upsample_count: usize,
}

impl<B: Backend> PPDocLayoutV3ScaleHead<B> {
    /// Loads one scale head and records how many 2x upsample steps it needs.
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        in_channels: usize,
        upsample_count: usize,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        let conv0 = ConvBnAct::load(
            weights,
            &format!("{prefix}.layers.0"),
            in_channels,
            64,
            3,
            1,
            1,
            Activation::Silu,
            device,
        )?;
        let conv1 = if upsample_count > 1 {
            Some(ConvBnAct::load(
                weights,
                &format!("{prefix}.layers.2"),
                64,
                64,
                3,
                1,
                1,
                Activation::Silu,
                device,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv0,
            conv1,
            upsample_count,
        })
    }

    /// Applies the scale head convolutions and requested bilinear upsampling steps.
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut output = self.conv0.forward(input);
        if self.upsample_count >= 2 {
            output = upsample_bilinear_2x(output);
            output = self.conv1.as_ref().unwrap().forward(output);
        }
        if self.upsample_count >= 1 {
            output = upsample_bilinear_2x(output);
        }
        output
    }
}

#[derive(Debug, Clone)]
struct PPDocLayoutV3EncoderMaskOutput<B: Backend> {
    base_conv: ConvBnAct<B>,
    conv: Conv2d<B>,
}

impl<B: Backend> PPDocLayoutV3EncoderMaskOutput<B> {
    /// Loads the final mask feature projection from 64 channels to 32 channels.
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        let mut conv = Conv2dConfig::new([64, 32], [1, 1])
            .with_bias(true)
            .init(device);
        conv.weight =
            Param::from_tensor(weights.tensor_f32(&format!("{prefix}.conv.weight"), device)?);
        conv.bias = Some(Param::from_tensor(
            weights.tensor_f32(&format!("{prefix}.conv.bias"), device)?,
        ));

        Ok(Self {
            base_conv: ConvBnAct::load(
                weights,
                &format!("{prefix}.base_conv"),
                64,
                64,
                3,
                1,
                1,
                Activation::Silu,
                device,
            )?,
            conv,
        })
    }

    /// Produces the encoder mask feature map consumed by decoder mask queries.
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(self.base_conv.forward(input))
    }
}

/// Upsamples a BCHW tensor by 2x using nearest-neighbor interpolation.
fn upsample_nearest_2x<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, height, width] = input.dims();
    interpolate(
        input,
        [height * 2, width * 2],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}

/// Upsamples a BCHW tensor by 2x using bilinear interpolation without aligned corners.
fn upsample_bilinear_2x<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, height, width] = input.dims();
    interpolate(
        input,
        [height * 2, width * 2],
        InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
    )
}

/// Builds sine/cosine 2D positional embeddings for flattened encoder tokens.
fn position_embeddings<B: Backend>(
    height: usize,
    width: usize,
    embed_dim: usize,
    device: B::Device,
) -> Tensor<B, 3> {
    let pos_dim = embed_dim / 4;
    let mut values = Vec::with_capacity(height * width * embed_dim);
    for h in 0..height {
        for w in 0..width {
            for i in 0..pos_dim {
                let omega = 1.0 / 10000.0_f32.powf(i as f32 / pos_dim as f32);
                values.push((h as f32 * omega).sin());
            }
            for i in 0..pos_dim {
                let omega = 1.0 / 10000.0_f32.powf(i as f32 / pos_dim as f32);
                values.push((h as f32 * omega).cos());
            }
            for i in 0..pos_dim {
                let omega = 1.0 / 10000.0_f32.powf(i as f32 / pos_dim as f32);
                values.push((w as f32 * omega).sin());
            }
            for i in 0..pos_dim {
                let omega = 1.0 / 10000.0_f32.powf(i as f32 / pos_dim as f32);
                values.push((w as f32 * omega).cos());
            }
        }
    }
    Tensor::from_data(
        TensorData::new(values, [1, height * width, embed_dim]),
        &device,
    )
}
