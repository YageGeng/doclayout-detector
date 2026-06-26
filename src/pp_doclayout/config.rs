use super::labels::PPDocLayoutV3Label;
use super::preprocess::PP_DOCLAYOUT_V3_IMAGE_SIZE;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PPDocLayoutV3Config {
    pub image_size: u32,
    pub num_queries: usize,
    pub num_classes: usize,
    pub d_model: usize,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub decoder_n_points: usize,
    pub encoder_in_channels: [usize; 3],
    pub decoder_in_channels: [usize; 3],
    pub feature_strides: [usize; 3],
    pub num_feature_levels: usize,
    pub global_pointer_head_size: usize,
    pub mask_feature_channels: [usize; 2],
    pub x4_feat_dim: usize,
}

impl Default for PPDocLayoutV3Config {
    /// Returns the PP-DocLayoutV3 architecture constants used by the loaded checkpoint.
    fn default() -> Self {
        Self {
            image_size: PP_DOCLAYOUT_V3_IMAGE_SIZE,
            num_queries: 300,
            num_classes: PPDocLayoutV3Label::class_count(),
            d_model: 256,
            decoder_layers: 6,
            decoder_attention_heads: 8,
            decoder_ffn_dim: 1024,
            decoder_n_points: 4,
            encoder_in_channels: [512, 1024, 2048],
            decoder_in_channels: [256, 256, 256],
            feature_strides: [8, 16, 32],
            num_feature_levels: 3,
            global_pointer_head_size: 64,
            mask_feature_channels: [64, 64],
            x4_feat_dim: 128,
        }
    }
}
