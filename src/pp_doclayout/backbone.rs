use super::layers::{Activation, ConvBnAct};
use super::weights::PPDocLayoutV3Weights;
use crate::error::LayoutError;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::ops::PadMode;
use burn_nn::pool::{MaxPool2d, MaxPool2dConfig};

#[derive(Debug, Clone)]
pub struct HgNetV2Stem<B: Backend> {
    stem1: ConvBnAct<B>,
    stem2a: ConvBnAct<B>,
    stem2b: ConvBnAct<B>,
    stem3: ConvBnAct<B>,
    stem4: ConvBnAct<B>,
    pool: MaxPool2d,
}

impl<B: Backend> HgNetV2Stem<B> {
    /// Loads the HGNetV2 stem layers that downsample RGB input into early feature maps.
    pub fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Ok(Self {
            stem1: ConvBnAct::load(
                weights,
                &format!("{prefix}.stem1"),
                3,
                32,
                3,
                2,
                1,
                Activation::Relu,
                device,
            )?,
            stem2a: ConvBnAct::load(
                weights,
                &format!("{prefix}.stem2a"),
                32,
                16,
                2,
                1,
                1,
                Activation::Relu,
                device,
            )?,
            stem2b: ConvBnAct::load(
                weights,
                &format!("{prefix}.stem2b"),
                16,
                32,
                2,
                1,
                1,
                Activation::Relu,
                device,
            )?,
            stem3: ConvBnAct::load(
                weights,
                &format!("{prefix}.stem3"),
                64,
                32,
                3,
                2,
                1,
                Activation::Relu,
                device,
            )?,
            stem4: ConvBnAct::load(
                weights,
                &format!("{prefix}.stem4"),
                32,
                48,
                1,
                1,
                1,
                Activation::Relu,
                device,
            )?,
            pool: MaxPool2dConfig::new([2, 2])
                .with_strides([1, 1])
                .with_ceil_mode(true)
                .init(),
        })
    }

    /// Runs the stem split-branch pooling path and returns the first backbone embedding.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let embedding = self
            .stem1
            .forward(input)
            .pad((0, 1, 0, 1), PadMode::Constant(0.0));
        let branch = self
            .stem2a
            .forward(embedding.clone())
            .pad((0, 1, 0, 1), PadMode::Constant(0.0));
        let branch = self.stem2b.forward(branch);
        let pooled = self.pool.forward(embedding);
        let embedding = Tensor::cat(vec![pooled, branch], 1);
        let embedding = self.stem3.forward(embedding);
        self.stem4.forward(embedding)
    }
}

#[derive(Debug, Clone)]
pub struct HgNetV2Backbone<B: Backend> {
    stem: HgNetV2Stem<B>,
    stages: Vec<HgNetV2Stage<B>>,
}

impl<B: Backend> HgNetV2Backbone<B> {
    /// Loads the full HGNetV2 backbone stages used by PP-DocLayoutV3.
    pub fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        let stem = HgNetV2Stem::load(weights, &format!("{prefix}.embedder"), device)?;
        let stages = vec![
            HgNetV2Stage::load(
                weights,
                &format!("{prefix}.encoder.stages.0"),
                HgNetV2StageConfig {
                    in_channels: 48,
                    mid_channels: 48,
                    out_channels: 128,
                    num_blocks: 1,
                    num_layers: 6,
                    downsample: false,
                    light_block: false,
                    kernel_size: 3,
                },
                device,
            )?,
            HgNetV2Stage::load(
                weights,
                &format!("{prefix}.encoder.stages.1"),
                HgNetV2StageConfig {
                    in_channels: 128,
                    mid_channels: 96,
                    out_channels: 512,
                    num_blocks: 1,
                    num_layers: 6,
                    downsample: true,
                    light_block: false,
                    kernel_size: 3,
                },
                device,
            )?,
            HgNetV2Stage::load(
                weights,
                &format!("{prefix}.encoder.stages.2"),
                HgNetV2StageConfig {
                    in_channels: 512,
                    mid_channels: 192,
                    out_channels: 1024,
                    num_blocks: 3,
                    num_layers: 6,
                    downsample: true,
                    light_block: true,
                    kernel_size: 5,
                },
                device,
            )?,
            HgNetV2Stage::load(
                weights,
                &format!("{prefix}.encoder.stages.3"),
                HgNetV2StageConfig {
                    in_channels: 1024,
                    mid_channels: 384,
                    out_channels: 2048,
                    num_blocks: 1,
                    num_layers: 6,
                    downsample: true,
                    light_block: true,
                    kernel_size: 5,
                },
                device,
            )?,
        ];

        Ok(Self { stem, stages })
    }

    /// Runs the backbone and returns the four stage feature maps used by the encoder.
    pub fn forward(&self, input: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let mut hidden = self.stem.forward(input);
        let mut features = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            hidden = stage.forward(hidden);
            features.push(hidden.clone());
        }
        features
    }
}

#[derive(Debug, Clone, Copy)]
struct HgNetV2StageConfig {
    in_channels: usize,
    mid_channels: usize,
    out_channels: usize,
    num_blocks: usize,
    num_layers: usize,
    downsample: bool,
    light_block: bool,
    kernel_size: usize,
}

#[derive(Debug, Clone)]
struct HgNetV2Stage<B: Backend> {
    downsample: Option<ConvBnAct<B>>,
    blocks: Vec<HgNetV2Block<B>>,
}

impl<B: Backend> HgNetV2Stage<B> {
    /// Loads one HGNetV2 stage, including optional depthwise downsampling.
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        config: HgNetV2StageConfig,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        let downsample = if config.downsample {
            Some(ConvBnAct::load(
                weights,
                &format!("{prefix}.downsample"),
                config.in_channels,
                config.in_channels,
                3,
                2,
                config.in_channels,
                Activation::None,
                device,
            )?)
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(config.num_blocks);
        for block_idx in 0..config.num_blocks {
            let in_channels = if block_idx == 0 {
                config.in_channels
            } else {
                config.out_channels
            };
            blocks.push(HgNetV2Block::load(
                weights,
                &format!("{prefix}.blocks.{block_idx}"),
                in_channels,
                config.mid_channels,
                config.out_channels,
                config.num_layers,
                config.kernel_size,
                config.light_block,
                block_idx > 0,
                device,
            )?);
        }

        Ok(Self { downsample, blocks })
    }

    /// Runs optional downsampling followed by all HGNetV2 blocks in this stage.
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut hidden = match &self.downsample {
            Some(downsample) => downsample.forward(input),
            None => input,
        };
        for block in &self.blocks {
            hidden = block.forward(hidden);
        }
        hidden
    }
}

#[derive(Debug, Clone)]
struct HgNetV2Block<B: Backend> {
    layers: Vec<HgNetV2Layer<B>>,
    aggregation0: ConvBnAct<B>,
    aggregation1: ConvBnAct<B>,
    residual: bool,
}

impl<B: Backend> HgNetV2Block<B> {
    #[allow(clippy::too_many_arguments)]
    /// Loads a dense HGNetV2 block with aggregation layers and optional residual output.
    fn load(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        in_channels: usize,
        mid_channels: usize,
        out_channels: usize,
        num_layers: usize,
        kernel_size: usize,
        light_block: bool,
        residual: bool,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let layer_in_channels = if layer_idx == 0 {
                in_channels
            } else {
                mid_channels
            };
            let layer = if light_block {
                HgNetV2Layer::load_light(
                    weights,
                    &format!("{prefix}.layers.{layer_idx}"),
                    layer_in_channels,
                    mid_channels,
                    kernel_size,
                    device,
                )?
            } else {
                HgNetV2Layer::load_regular(
                    weights,
                    &format!("{prefix}.layers.{layer_idx}"),
                    layer_in_channels,
                    mid_channels,
                    kernel_size,
                    device,
                )?
            };
            layers.push(layer);
        }

        let concat_channels = in_channels + num_layers * mid_channels;
        let aggregation0 = ConvBnAct::load(
            weights,
            &format!("{prefix}.aggregation.0"),
            concat_channels,
            out_channels / 2,
            1,
            1,
            1,
            Activation::Relu,
            device,
        )?;
        let aggregation1 = ConvBnAct::load(
            weights,
            &format!("{prefix}.aggregation.1"),
            out_channels / 2,
            out_channels,
            1,
            1,
            1,
            Activation::Relu,
            device,
        )?;

        Ok(Self {
            layers,
            aggregation0,
            aggregation1,
            residual,
        })
    }

    /// Runs all internal layers, concatenates intermediate states, and aggregates channels.
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = input.clone();
        let mut hidden = input;
        let mut states = vec![hidden.clone()];
        for layer in &self.layers {
            hidden = layer.forward(hidden);
            states.push(hidden.clone());
        }
        let hidden = Tensor::cat(states, 1);
        let hidden = self.aggregation0.forward(hidden);
        let hidden = self.aggregation1.forward(hidden);
        if self.residual {
            hidden + identity
        } else {
            hidden
        }
    }
}

#[derive(Debug, Clone)]
enum HgNetV2Layer<B: Backend> {
    Regular(Box<ConvBnAct<B>>),
    Light {
        conv1: Box<ConvBnAct<B>>,
        conv2: Box<ConvBnAct<B>>,
    },
}

impl<B: Backend> HgNetV2Layer<B> {
    /// Loads a regular HGNetV2 convolutional layer.
    fn load_regular(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Ok(Self::Regular(Box::new(ConvBnAct::load(
            weights,
            prefix,
            in_channels,
            out_channels,
            kernel_size,
            1,
            1,
            Activation::Relu,
            device,
        )?)))
    }

    /// Loads a lightweight HGNetV2 layer with pointwise and depthwise convolutions.
    fn load_light(
        weights: &PPDocLayoutV3Weights,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        device: &B::Device,
    ) -> Result<Self, LayoutError> {
        Ok(Self::Light {
            conv1: Box::new(ConvBnAct::load(
                weights,
                &format!("{prefix}.conv1"),
                in_channels,
                out_channels,
                1,
                1,
                1,
                Activation::None,
                device,
            )?),
            conv2: Box::new(ConvBnAct::load(
                weights,
                &format!("{prefix}.conv2"),
                out_channels,
                out_channels,
                kernel_size,
                1,
                out_channels,
                Activation::Relu,
                device,
            )?),
        })
    }

    /// Runs either the regular layer or the two-step lightweight layer.
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::Regular(layer) => layer.forward(input),
            Self::Light { conv1, conv2 } => conv2.forward(conv1.forward(input)),
        }
    }
}
