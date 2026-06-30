use super::weights::PPDocLayoutV3Weights;
use crate::error::LayoutError;
use burn::module::Param;
use burn::tensor::activation::{relu, silu};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_nn::conv::{Conv2d, Conv2dConfig};
use burn_nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};

const BATCH_NORM_EPSILON: f64 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    None,
    Relu,
    Silu,
}

impl Activation {
    /// Applies the configured activation without changing tensor rank or layout.
    fn forward<B: Backend, const D: usize>(self, input: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Self::None => input,
            Self::Relu => relu(input),
            Self::Silu => silu(input),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConvBnAct<B: Backend> {
    conv: Conv2d<B>,
    activation: Activation,
}

#[derive(Debug, Clone)]
pub struct ConvNormAct<B: Backend> {
    conv: Conv2d<B>,
    activation: Activation,
}

impl<B: Backend> ConvNormAct<B> {
    #[allow(clippy::too_many_arguments)]
    /// Loads a convolution, batch normalization, and activation block with explicit submodule names.
    pub fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        conv_name: &str,
        norm_name: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        activation: Activation,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        let padding = (kernel_size - 1) / 2;
        let mut conv = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
            .with_stride([stride, stride])
            .with_padding(burn_nn::PaddingConfig2d::Explicit(
                padding, padding, padding, padding,
            ))
            .with_bias(true)
            .init(device);
        let (conv_weight, conv_shape) =
            weights.tensor_f32_values::<4>(&format!("{prefix}.{conv_name}.weight"))?;
        ensure_conv_out_channels(prefix, conv_shape[0], out_channels)?;
        let (gamma, _) = weights.tensor_f32_values::<1>(&format!("{prefix}.{norm_name}.weight"))?;
        let (beta, _) = weights.tensor_f32_values::<1>(&format!("{prefix}.{norm_name}.bias"))?;
        let (running_mean, _) =
            weights.tensor_f32_values::<1>(&format!("{prefix}.{norm_name}.running_mean"))?;
        let (running_var, _) =
            weights.tensor_f32_values::<1>(&format!("{prefix}.{norm_name}.running_var"))?;
        let (weight, bias) = fold_batch_norm_into_conv_values(
            conv_weight,
            conv_shape[0],
            &gamma,
            &beta,
            &running_mean,
            &running_var,
            BATCH_NORM_EPSILON,
        )?;
        conv.weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(weight, conv_shape),
            device,
        ));
        conv.bias = Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(bias, [out_channels]),
            device,
        )));

        Ok(Self { conv, activation })
    }

    /// Runs the batchnorm-folded convolution and activation for a BCHW feature map.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.activation.forward(self.conv.forward(input))
    }
}

impl<B: Backend> ConvBnAct<B> {
    #[allow(clippy::too_many_arguments)]
    /// Loads a convolution-batchnorm-activation block using default same-padding.
    pub fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        activation: Activation,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Self::load_with_padding(
            weights,
            prefix,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            (kernel_size - 1) / 2,
            activation,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Loads a convolution-batchnorm-activation block with caller-supplied padding.
    pub fn load_with_padding(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        padding: usize,
        activation: Activation,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        let mut conv = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
            .with_stride([stride, stride])
            .with_padding(burn_nn::PaddingConfig2d::Explicit(
                padding, padding, padding, padding,
            ))
            .with_groups(groups)
            .with_bias(true)
            .init(device);
        let (conv_weight, conv_shape) =
            weights.tensor_f32_values::<4>(&format!("{prefix}.convolution.weight"))?;
        ensure_conv_out_channels(prefix, conv_shape[0], out_channels)?;
        let (gamma, _) =
            weights.tensor_f32_values::<1>(&format!("{prefix}.normalization.weight"))?;
        let (beta, _) = weights.tensor_f32_values::<1>(&format!("{prefix}.normalization.bias"))?;
        let (running_mean, _) =
            weights.tensor_f32_values::<1>(&format!("{prefix}.normalization.running_mean"))?;
        let (running_var, _) =
            weights.tensor_f32_values::<1>(&format!("{prefix}.normalization.running_var"))?;
        let (weight, bias) = fold_batch_norm_into_conv_values(
            conv_weight,
            conv_shape[0],
            &gamma,
            &beta,
            &running_mean,
            &running_var,
            BATCH_NORM_EPSILON,
        )?;
        conv.weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(weight, conv_shape),
            device,
        ));
        conv.bias = Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(bias, [out_channels]),
            device,
        )));

        Ok(Self { conv, activation })
    }

    /// Runs the batchnorm-folded convolution and activation for a BCHW feature map.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.activation.forward(self.conv.forward(input))
    }
}

/// Folds inference BatchNorm parameters into convolution weight and bias values.
fn fold_batch_norm_into_conv_values(
    mut conv_weight: Vec<f32>,
    out_channels: usize,
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    epsilon: f64,
) -> Result<(Vec<f32>, Vec<f32>), LayoutError> {
    if out_channels == 0 || conv_weight.len() % out_channels != 0 {
        return Err(LayoutError::InvalidModelOutput(
            "convolution weight shape is incompatible with batch norm".to_string(),
        ));
    }
    if gamma.len() != out_channels
        || beta.len() != out_channels
        || running_mean.len() != out_channels
        || running_var.len() != out_channels
    {
        return Err(LayoutError::InvalidModelOutput(format!(
            "batch norm parameter length mismatch for {out_channels} output channels"
        )));
    }

    let values_per_channel = conv_weight.len() / out_channels;
    let mut bias = Vec::with_capacity(out_channels);
    for channel in 0..out_channels {
        let scale = gamma[channel] / (running_var[channel] + epsilon as f32).sqrt();
        let start = channel * values_per_channel;
        let end = start + values_per_channel;
        for value in &mut conv_weight[start..end] {
            *value *= scale;
        }
        bias.push(beta[channel] - running_mean[channel] * scale);
    }

    Ok((conv_weight, bias))
}

fn ensure_conv_out_channels(
    prefix: &str,
    actual: usize,
    expected: usize,
) -> Result<(), LayoutError> {
    if actual != expected {
        return Err(LayoutError::InvalidModelOutput(format!(
            "expected convolution {prefix} to have {expected} output channels, got {actual}"
        )));
    }
    Ok(())
}

/// Loads a Paddle-style linear layer and transposes weights for Burn's layout.
pub fn load_linear<B: Backend>(
    weights: &PPDocLayoutV3Weights,
    prefix: &str,
    d_input: usize,
    d_output: usize,
    bias: bool,
    device: &B::Device,
) -> Result<Linear<B>, LayoutError> {
    let mut linear = LinearConfig::new(d_input, d_output)
        .with_bias(bias)
        .init(device);
    let weight = weights
        .tensor_f32::<B, 2>(&format!("{prefix}.weight"), device)?
        .transpose();
    linear.weight = Param::from_tensor(weight);
    if bias {
        linear.bias = Some(Param::from_tensor(
            weights.tensor_f32(&format!("{prefix}.bias"), device)?,
        ));
    }
    Ok(linear)
}

/// Loads a layer normalization module from PP-DocLayoutV3 checkpoint tensors.
pub fn load_layer_norm<B: Backend>(
    weights: &PPDocLayoutV3Weights,
    prefix: &str,
    d_model: usize,
    device: &B::Device,
) -> Result<LayerNorm<B>, LayoutError> {
    let mut norm = LayerNormConfig::new(d_model).init(device);
    norm.gamma = Param::from_tensor(weights.tensor_f32(&format!("{prefix}.weight"), device)?);
    norm.beta = Some(Param::from_tensor(
        weights.tensor_f32(&format!("{prefix}.bias"), device)?,
    ));
    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_batch_norm_into_conv_values_matches_eval_formula() {
        let (weight_values, bias_values) = fold_batch_norm_into_conv_values(
            vec![2.0, -4.0, 1.0, 3.0],
            2,
            &[3.0, 0.5],
            &[1.0, -2.0],
            &[4.0, -1.0],
            &[8.0, 3.0],
            1.0,
        )
        .unwrap();

        assert_close(&weight_values, &[2.0, -4.0, 0.25, 0.75]);
        assert_close(&bias_values, &[-3.0, -1.75]);
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "expected {expected}, got {actual}"
            );
        }
    }
}
