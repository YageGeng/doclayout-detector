use doclayout_detector::pp_doclayout::{
    Activation, ConvBnAct, HgNetV2Backbone, HgNetV2Stem, PPDocLayoutV3EncoderInputProjection,
    PPDocLayoutV3HybridEncoder, PPDocLayoutV3Weights,
};
use std::path::Path;
use std::sync::Once;

type TestBackend = doclayout_detector::model::LayoutBackend;

static INIT_WGPU: Once = Once::new();

/// Returns the shared test device after one-time Burn WGPU initialization.
fn test_device() -> burn_wgpu::WgpuDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    INIT_WGPU.call_once(|| init_wgpu_test_backend(&device));
    device
}

fn init_wgpu_test_backend(device: &burn_wgpu::WgpuDevice) {
    #[cfg(feature = "backend-metal")]
    burn_wgpu::init_setup::<burn_wgpu::graphics::Metal>(device, Default::default());
    #[cfg(feature = "backend-vulkan")]
    burn_wgpu::init_setup::<burn_wgpu::graphics::Vulkan>(device, Default::default());
    #[cfg(feature = "backend-webgpu")]
    burn_wgpu::init_setup::<burn_wgpu::graphics::WebGpu>(device, Default::default());
}

#[test]
fn pp_doclayout_conv_bn_act_loads_real_stem_weight_and_forwards() {
    let path = Path::new("models/pp_doclayout_v3/model.safetensors");
    if !path.exists() {
        tracing::warn!(path = %path.display(), "skipping PP layer test; model file is absent");
        return;
    }

    let device = test_device();
    let weights = PPDocLayoutV3Weights::from_file(path).unwrap();
    let layer = ConvBnAct::<TestBackend>::load(
        &weights,
        "model.backbone.model.embedder.stem1",
        3,
        32,
        3,
        2,
        1,
        Activation::Relu,
        &device,
    )
    .unwrap();
    let input = burn::tensor::Tensor::<TestBackend, 4>::zeros([1, 3, 16, 16], &device);

    let output = layer.forward(input);

    assert_eq!(output.dims(), [1, 32, 8, 8]);
}

#[test]
fn pp_doclayout_hgnet_stem_loads_real_weights_and_forwards() {
    let path = Path::new("models/pp_doclayout_v3/model.safetensors");
    if !path.exists() {
        tracing::warn!(path = %path.display(), "skipping PP stem test; model file is absent");
        return;
    }

    let device = test_device();
    let weights = PPDocLayoutV3Weights::from_file(path).unwrap();
    let stem = HgNetV2Stem::<TestBackend>::load(&weights, "model.backbone.model.embedder", &device)
        .unwrap();
    let input = burn::tensor::Tensor::<TestBackend, 4>::zeros([1, 3, 64, 64], &device);

    let output = stem.forward(input);

    assert_eq!(output.dims(), [1, 48, 16, 16]);
}

#[test]
fn pp_doclayout_hgnet_backbone_loads_real_weights_and_forwards() {
    let path = Path::new("models/pp_doclayout_v3/model.safetensors");
    if !path.exists() {
        tracing::warn!(path = %path.display(), "skipping PP backbone test; model file is absent");
        return;
    }

    let device = test_device();
    let weights = PPDocLayoutV3Weights::from_file(path).unwrap();
    let backbone =
        HgNetV2Backbone::<TestBackend>::load(&weights, "model.backbone.model", &device).unwrap();
    let input = burn::tensor::Tensor::<TestBackend, 4>::zeros([1, 3, 64, 64], &device);

    let features = backbone.forward(input);

    assert_eq!(features.len(), 4);
    assert_eq!(features[0].dims(), [1, 128, 16, 16]);
    assert_eq!(features[1].dims(), [1, 512, 8, 8]);
    assert_eq!(features[2].dims(), [1, 1024, 4, 4]);
    assert_eq!(features[3].dims(), [1, 2048, 2, 2]);
}

#[test]
fn pp_doclayout_encoder_input_projection_loads_real_weights_and_forwards() {
    let path = Path::new("models/pp_doclayout_v3/model.safetensors");
    if !path.exists() {
        tracing::warn!(path = %path.display(), "skipping PP projection test; model file is absent");
        return;
    }

    let device = test_device();
    let weights = PPDocLayoutV3Weights::from_file(path).unwrap();
    let backbone =
        HgNetV2Backbone::<TestBackend>::load(&weights, "model.backbone.model", &device).unwrap();
    let projection = PPDocLayoutV3EncoderInputProjection::<TestBackend>::load(
        &weights,
        "model.encoder_input_proj",
        &device,
    )
    .unwrap();
    let input = burn::tensor::Tensor::<TestBackend, 4>::zeros([1, 3, 64, 64], &device);

    let features = backbone.forward(input);
    let projected = projection.forward(vec![
        features[1].clone(),
        features[2].clone(),
        features[3].clone(),
    ]);

    assert_eq!(projected.len(), 3);
    assert_eq!(projected[0].dims(), [1, 256, 8, 8]);
    assert_eq!(projected[1].dims(), [1, 256, 4, 4]);
    assert_eq!(projected[2].dims(), [1, 256, 2, 2]);
}

#[test]
fn pp_doclayout_hybrid_encoder_loads_real_weights_and_forwards() {
    let path = Path::new("models/pp_doclayout_v3/model.safetensors");
    if !path.exists() {
        tracing::warn!(path = %path.display(), "skipping PP hybrid encoder test; model file is absent");
        return;
    }

    let device = test_device();
    let weights = PPDocLayoutV3Weights::from_file(path).unwrap();
    let backbone =
        HgNetV2Backbone::<TestBackend>::load(&weights, "model.backbone.model", &device).unwrap();
    let projection = PPDocLayoutV3EncoderInputProjection::<TestBackend>::load(
        &weights,
        "model.encoder_input_proj",
        &device,
    )
    .unwrap();
    let encoder =
        PPDocLayoutV3HybridEncoder::<TestBackend>::load(&weights, "model.encoder", &device)
            .unwrap();
    let input = burn::tensor::Tensor::<TestBackend, 4>::zeros([1, 3, 64, 64], &device);

    let features = backbone.forward(input);
    let projected = projection.forward(vec![
        features[1].clone(),
        features[2].clone(),
        features[3].clone(),
    ]);
    let output = encoder.forward(projected, vec![features[0].clone()]);

    assert_eq!(output.last_hidden_state.len(), 3);
    assert_eq!(output.last_hidden_state[0].dims(), [1, 256, 8, 8]);
    assert_eq!(output.last_hidden_state[1].dims(), [1, 256, 4, 4]);
    assert_eq!(output.last_hidden_state[2].dims(), [1, 256, 2, 2]);
    assert_eq!(output.mask_feat.dims(), [1, 32, 16, 16]);
}
