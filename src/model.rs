use crate::error::LayoutError;
use crate::pp_doclayout::{
    PP_DOCLAYOUT_V3_IMAGE_SIZE, PPDocLayoutV3Inference, PPDocLayoutV3Model,
    PPDocLayoutV3OwnedOutputs, PPDocLayoutV3Weights,
};
use burn::tensor::{Tensor, TensorData};
#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
use tracing::Level;

#[cfg(not(any(
    feature = "backend-metal",
    feature = "backend-vulkan",
    feature = "backend-webgpu"
)))]
compile_error!(
    "select one layout backend feature: backend-metal, backend-vulkan, or backend-webgpu"
);

#[cfg(all(
    feature = "backend-metal",
    not(any(feature = "backend-vulkan", feature = "backend-webgpu"))
))]
pub type LayoutBackend = burn_wgpu::Metal;
#[cfg(all(
    feature = "backend-metal",
    not(any(feature = "backend-vulkan", feature = "backend-webgpu"))
))]
pub type LayoutDevice = burn_wgpu::WgpuDevice;

#[cfg(all(feature = "backend-vulkan", not(feature = "backend-webgpu")))]
pub type LayoutBackend = burn_wgpu::Vulkan;
#[cfg(all(feature = "backend-vulkan", not(feature = "backend-webgpu")))]
pub type LayoutDevice = burn_wgpu::WgpuDevice;

#[cfg(feature = "backend-webgpu")]
pub type LayoutBackend = burn_wgpu::WebGpu;
#[cfg(feature = "backend-webgpu")]
pub type LayoutDevice = burn_wgpu::WgpuDevice;

#[cfg(all(
    feature = "backend-metal",
    not(any(feature = "backend-vulkan", feature = "backend-webgpu"))
))]
const BACKEND_NAME: &str = "metal";
#[cfg(all(feature = "backend-vulkan", not(feature = "backend-webgpu")))]
const BACKEND_NAME: &str = "vulkan";
#[cfg(feature = "backend-webgpu")]
const BACKEND_NAME: &str = "webgpu";

#[derive(Debug, Clone)]
pub struct EmbeddedModel {
    device: LayoutDevice,
    model: PPDocLayoutV3Model<LayoutBackend>,
}

impl EmbeddedModel {
    #[cfg(not(all(target_family = "wasm", feature = "backend-webgpu")))]
    /// Initializes the compiled native backend and loads the embedded layout model.
    pub fn new() -> Result<Self, LayoutError> {
        let device = create_device();
        tracing::info!(backend = BACKEND_NAME, "initialized layout backend");
        let model = load_model(&device)?;
        Ok(Self { device, model })
    }

    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    /// Initializes the browser WebGPU backend asynchronously and loads the embedded layout model.
    pub async fn new_async() -> Result<Self, LayoutError> {
        let device = create_device_async().await;
        tracing::info!(backend = BACKEND_NAME, "initialized layout backend");
        let model = load_model(&device)?;
        Ok(Self { device, model })
    }
}

impl PPDocLayoutV3Inference for EmbeddedModel {
    /// Runs synchronous model inference and copies tensor outputs back to owned host buffers.
    fn infer(&self, input: &[f32]) -> Result<PPDocLayoutV3OwnedOutputs, LayoutError> {
        self.infer_batch(input, 1)
    }

    /// Runs synchronous batched inference and copies tensor outputs back to owned host buffers.
    fn infer_batch(
        &self,
        input: &[f32],
        batch_size: usize,
    ) -> Result<PPDocLayoutV3OwnedOutputs, LayoutError> {
        if batch_size == 0 {
            return Err(LayoutError::InvalidModelOutput(
                "batch size must be greater than zero".to_string(),
            ));
        }
        let expected = batch_size
            * 3
            * PP_DOCLAYOUT_V3_IMAGE_SIZE as usize
            * PP_DOCLAYOUT_V3_IMAGE_SIZE as usize;
        if input.len() != expected {
            return Err(LayoutError::InvalidModelOutput(format!(
                "expected batched CHW input length {expected}, got {}",
                input.len()
            )));
        }

        // The model supports a dynamic batch dimension while keeping the fixed
        // official page shape of 3x800x800 for every item in the batch.
        let tensor = Tensor::<LayoutBackend, 4>::from_data(
            TensorData::new(
                input.to_vec(),
                [
                    batch_size,
                    3,
                    PP_DOCLAYOUT_V3_IMAGE_SIZE as usize,
                    PP_DOCLAYOUT_V3_IMAGE_SIZE as usize,
                ],
            ),
            &self.device,
        );
        let output = self.model.forward(tensor);
        let logits_shape = output.logits.dims();
        let pred_boxes_shape = output.pred_boxes.dims();
        let order_logits_shape = output.order_logits.as_ref().map(|tensor| tensor.dims());
        let logits = output.logits.into_data().to_vec::<f32>().map_err(|error| {
            LayoutError::InvalidModelOutput(format!("read logits tensor: {error}"))
        })?;
        let pred_boxes = output
            .pred_boxes
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| {
                LayoutError::InvalidModelOutput(format!("read pred boxes tensor: {error}"))
            })?;
        let order_logits = output
            .order_logits
            .map(|tensor| {
                tensor.into_data().to_vec::<f32>().map_err(|error| {
                    LayoutError::InvalidModelOutput(format!("read order logits tensor: {error}"))
                })
            })
            .transpose()?;

        Ok(PPDocLayoutV3OwnedOutputs {
            logits_shape,
            logits,
            pred_boxes_shape,
            pred_boxes,
            order_logits_shape,
            order_logits,
        })
    }
}

#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
impl EmbeddedModel {
    /// Runs browser WebGPU inference with async tensor readbacks for postprocessing.
    pub async fn infer_async(
        &self,
        input: &[f32],
    ) -> Result<PPDocLayoutV3OwnedOutputs, LayoutError> {
        self.infer_batch_async(input, 1).await
    }

    /// Runs browser WebGPU batched inference with async tensor readbacks.
    pub async fn infer_batch_async(
        &self,
        input: &[f32],
        batch_size: usize,
    ) -> Result<PPDocLayoutV3OwnedOutputs, LayoutError> {
        if batch_size == 0 {
            return Err(LayoutError::InvalidModelOutput(
                "batch size must be greater than zero".to_string(),
            ));
        }
        let expected = batch_size
            * 3
            * PP_DOCLAYOUT_V3_IMAGE_SIZE as usize
            * PP_DOCLAYOUT_V3_IMAGE_SIZE as usize;
        if input.len() != expected {
            return Err(LayoutError::InvalidModelOutput(format!(
                "expected batched CHW input length {expected}, got {}",
                input.len()
            )));
        }

        let upload_profile = EmbeddedWasmTimer::start("input_upload");
        let tensor = Tensor::<LayoutBackend, 4>::from_data(
            TensorData::new(
                input.to_vec(),
                [
                    batch_size,
                    3,
                    PP_DOCLAYOUT_V3_IMAGE_SIZE as usize,
                    PP_DOCLAYOUT_V3_IMAGE_SIZE as usize,
                ],
            ),
            &self.device,
        );
        upload_profile.finish();

        let forward_profile = EmbeddedWasmTimer::start("model_forward_async");
        let output = self.model.forward_async(tensor).await?;
        forward_profile.finish();
        let logits_shape = output.logits.dims();
        let pred_boxes_shape = output.pred_boxes.dims();
        let order_logits_shape = output.order_logits.as_ref().map(|tensor| tensor.dims());

        let logits_profile = EmbeddedWasmTimer::start("logits_readback");
        let logits = output
            .logits
            .into_data_async()
            .await
            .map_err(|error| {
                LayoutError::InvalidModelOutput(format!("read logits tensor: {error}"))
            })?
            .to_vec::<f32>()
            .map_err(|error| {
                LayoutError::InvalidModelOutput(format!("decode logits tensor: {error}"))
            })?;
        logits_profile.finish();

        let boxes_profile = EmbeddedWasmTimer::start("pred_boxes_readback");
        let pred_boxes = output
            .pred_boxes
            .into_data_async()
            .await
            .map_err(|error| {
                LayoutError::InvalidModelOutput(format!("read pred boxes tensor: {error}"))
            })?
            .to_vec::<f32>()
            .map_err(|error| {
                LayoutError::InvalidModelOutput(format!("decode pred boxes tensor: {error}"))
            })?;
        boxes_profile.finish();

        let order_logits = match output.order_logits {
            Some(tensor) => {
                let order_profile = EmbeddedWasmTimer::start("order_logits_readback");
                let values = tensor
                    .into_data_async()
                    .await
                    .map_err(|error| {
                        LayoutError::InvalidModelOutput(format!(
                            "read order logits tensor: {error}"
                        ))
                    })?
                    .to_vec::<f32>()
                    .map_err(|error| {
                        LayoutError::InvalidModelOutput(format!(
                            "decode order logits tensor: {error}"
                        ))
                    })?;
                order_profile.finish();
                Some(values)
            }
            None => None,
        };

        Ok(PPDocLayoutV3OwnedOutputs {
            logits_shape,
            logits,
            pred_boxes_shape,
            pred_boxes,
            order_logits_shape,
            order_logits,
        })
    }
}

#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
#[derive(Debug, Clone)]
struct EmbeddedWasmTimer {
    step: &'static str,
    started_ms: f64,
}

#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
impl EmbeddedWasmTimer {
    /// Start a browser-compatible timer for embedded model IO steps.
    fn start(step: &'static str) -> Self {
        Self {
            step,
            started_ms: js_sys::Date::now(),
        }
    }

    /// Emit a tracing event with elapsed milliseconds for this model IO step.
    fn finish(self) {
        tracing::event!(
            Level::INFO,
            step = self.step,
            duration_ms = js_sys::Date::now() - self.started_ms,
            "embedded model step completed"
        );
    }
}

/// Loads the embedded PP-DocLayoutV3 safetensors weights into the selected backend.
fn load_model(device: &LayoutDevice) -> Result<PPDocLayoutV3Model<LayoutBackend>, LayoutError> {
    let weights = PPDocLayoutV3Weights::from_bytes(
        include_bytes!("../models/pp_doclayout_v3/model.safetensors").to_vec(),
    )?;
    PPDocLayoutV3Model::load_weights(&weights, device)
}

#[cfg(all(
    feature = "backend-metal",
    not(any(feature = "backend-vulkan", feature = "backend-webgpu"))
))]
/// Creates and initializes a native Metal-backed WGPU device.
fn create_device() -> LayoutDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup::<burn_wgpu::graphics::Metal>(&device, Default::default());
    device
}

#[cfg(all(feature = "backend-vulkan", not(feature = "backend-webgpu")))]
/// Creates and initializes a native Vulkan-backed WGPU device.
fn create_device() -> LayoutDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup::<burn_wgpu::graphics::Vulkan>(&device, Default::default());
    device
}

#[cfg(all(not(target_family = "wasm"), feature = "backend-webgpu"))]
/// Creates and initializes a native WebGPU-backed WGPU device.
fn create_device() -> LayoutDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup::<burn_wgpu::graphics::WebGpu>(&device, Default::default());
    device
}

#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
/// Creates and initializes a browser WebGPU device using Burn's async setup path.
async fn create_device_async() -> LayoutDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup_async::<burn_wgpu::graphics::WebGpu>(&device, Default::default()).await;
    device
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_compiled_backend_name() {
        assert!(matches!(BACKEND_NAME, "metal" | "vulkan" | "webgpu"));
    }
}
