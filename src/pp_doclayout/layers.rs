use super::weights::PPDocLayoutV3Weights;
use crate::error::LayoutError;
use burn::module::{Param, RunningState};
use burn::tensor::Tensor;
use burn::tensor::activation::{relu, silu};
use burn::tensor::backend::Backend;
use burn_nn::conv::{Conv2d, Conv2dConfig};
use burn_nn::{BatchNorm, BatchNormConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};

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
    norm: BatchNorm<B>,
    activation: Activation,
}

#[derive(Debug, Clone)]
pub struct ConvNormAct<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B>,
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
            .with_bias(false)
            .init(device);
        conv.weight = Param::from_tensor(
            weights.tensor_f32(&format!("{prefix}.{conv_name}.weight"), device)?,
        );

        let mut norm = BatchNormConfig::new(out_channels)
            .with_epsilon(1e-5)
            .init(device);
        norm.gamma = Param::from_tensor(
            weights.tensor_f32(&format!("{prefix}.{norm_name}.weight"), device)?,
        );
        norm.beta =
            Param::from_tensor(weights.tensor_f32(&format!("{prefix}.{norm_name}.bias"), device)?);
        norm.running_mean = RunningState::new(
            weights.tensor_f32(&format!("{prefix}.{norm_name}.running_mean"), device)?,
        );
        norm.running_var = RunningState::new(
            weights.tensor_f32(&format!("{prefix}.{norm_name}.running_var"), device)?,
        );

        Ok(Self {
            conv,
            norm,
            activation,
        })
    }

    /// Runs convolution, batch normalization, and activation for a BCHW feature map.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.conv.forward(input);
        let output = self.norm.forward(output);
        self.activation.forward(output)
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
            .with_bias(false)
            .init(device);
        conv.weight = Param::from_tensor(
            weights.tensor_f32(&format!("{prefix}.convolution.weight"), device)?,
        );

        let mut norm = BatchNormConfig::new(out_channels).init(device);
        norm.gamma = Param::from_tensor(
            weights.tensor_f32(&format!("{prefix}.normalization.weight"), device)?,
        );
        norm.beta = Param::from_tensor(
            weights.tensor_f32(&format!("{prefix}.normalization.bias"), device)?,
        );
        norm.running_mean = RunningState::new(
            weights.tensor_f32(&format!("{prefix}.normalization.running_mean"), device)?,
        );
        norm.running_var = RunningState::new(
            weights.tensor_f32(&format!("{prefix}.normalization.running_var"), device)?,
        );

        Ok(Self {
            conv,
            norm,
            activation,
        })
    }

    /// Runs convolution, batch normalization, and activation for a BCHW feature map.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.conv.forward(input);
        let output = self.norm.forward(output);
        self.activation.forward(output)
    }
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
