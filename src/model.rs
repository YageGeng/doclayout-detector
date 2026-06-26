use crate::error::LayoutError;
use crate::pp_doclayout::{
    PP_DOCLAYOUT_V3_IMAGE_SIZE, PPDocLayoutV3Inference, PPDocLayoutV3Model,
    PPDocLayoutV3OwnedOutputs, PPDocLayoutV3Weights,
};
use burn::tensor::{Tensor, TensorData};

#[cfg(not(any(
    feature = "backend-ndarray",
    feature = "backend-metal",
    feature = "backend-vulkan",
    feature = "backend-webgpu"
)))]
compile_error!(
    "select one layout backend feature: backend-ndarray, backend-metal, backend-vulkan, or backend-webgpu"
);

#[cfg(all(
    feature = "backend-ndarray",
    not(any(
        feature = "backend-metal",
        feature = "backend-vulkan",
        feature = "backend-webgpu"
    ))
))]
pub type LayoutBackend = burn_ndarray::NdArray<f32>;
#[cfg(all(
    feature = "backend-ndarray",
    not(any(
        feature = "backend-metal",
        feature = "backend-vulkan",
        feature = "backend-webgpu"
    ))
))]
pub type LayoutDevice = burn_ndarray::NdArrayDevice;

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
    feature = "backend-ndarray",
    not(any(
        feature = "backend-metal",
        feature = "backend-vulkan",
        feature = "backend-webgpu"
    ))
))]
const BACKEND_NAME: &str = "ndarray";
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
    pub fn new() -> Result<Self, LayoutError> {
        let device = create_device();
        tracing::info!(backend = BACKEND_NAME, "initialized layout backend");
        let model = load_model(&device)?;
        Ok(Self { device, model })
    }

    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    pub async fn new_async() -> Result<Self, LayoutError> {
        let device = create_device_async().await;
        tracing::info!(backend = BACKEND_NAME, "initialized layout backend");
        let model = load_model(&device)?;
        Ok(Self { device, model })
    }
}

impl PPDocLayoutV3Inference for EmbeddedModel {
    fn infer(&self, input: &[f32]) -> Result<PPDocLayoutV3OwnedOutputs, LayoutError> {
        let expected =
            3 * PP_DOCLAYOUT_V3_IMAGE_SIZE as usize * PP_DOCLAYOUT_V3_IMAGE_SIZE as usize;
        if input.len() != expected {
            return Err(LayoutError::InvalidModelOutput(format!(
                "expected CHW input length {expected}, got {}",
                input.len()
            )));
        }

        let tensor = Tensor::<LayoutBackend, 4>::from_data(
            TensorData::new(
                input.to_vec(),
                [
                    1,
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
    pub async fn infer_async(
        &self,
        input: &[f32],
    ) -> Result<PPDocLayoutV3OwnedOutputs, LayoutError> {
        let expected =
            3 * PP_DOCLAYOUT_V3_IMAGE_SIZE as usize * PP_DOCLAYOUT_V3_IMAGE_SIZE as usize;
        if input.len() != expected {
            return Err(LayoutError::InvalidModelOutput(format!(
                "expected CHW input length {expected}, got {}",
                input.len()
            )));
        }

        let tensor = Tensor::<LayoutBackend, 4>::from_data(
            TensorData::new(
                input.to_vec(),
                [
                    1,
                    3,
                    PP_DOCLAYOUT_V3_IMAGE_SIZE as usize,
                    PP_DOCLAYOUT_V3_IMAGE_SIZE as usize,
                ],
            ),
            &self.device,
        );
        let output = self.model.forward_async(tensor).await?;
        let logits_shape = output.logits.dims();
        let pred_boxes_shape = output.pred_boxes.dims();
        let order_logits_shape = output.order_logits.as_ref().map(|tensor| tensor.dims());
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
        let order_logits = match output.order_logits {
            Some(tensor) => Some(
                tensor
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
                    })?,
            ),
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

fn load_model(device: &LayoutDevice) -> Result<PPDocLayoutV3Model<LayoutBackend>, LayoutError> {
    let weights = PPDocLayoutV3Weights::from_bytes(
        include_bytes!("../models/pp_doclayout_v3/model.safetensors").to_vec(),
    )?;
    PPDocLayoutV3Model::load_weights(&weights, device)
}

#[cfg(all(
    feature = "backend-ndarray",
    not(any(
        feature = "backend-metal",
        feature = "backend-vulkan",
        feature = "backend-webgpu"
    ))
))]
fn create_device() -> LayoutDevice {
    burn_ndarray::NdArrayDevice::Cpu
}

#[cfg(all(
    feature = "backend-metal",
    not(any(feature = "backend-vulkan", feature = "backend-webgpu"))
))]
fn create_device() -> LayoutDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup::<burn_wgpu::graphics::Metal>(&device, Default::default());
    device
}

#[cfg(all(feature = "backend-vulkan", not(feature = "backend-webgpu")))]
fn create_device() -> LayoutDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup::<burn_wgpu::graphics::Vulkan>(&device, Default::default());
    device
}

#[cfg(all(not(target_family = "wasm"), feature = "backend-webgpu"))]
fn create_device() -> LayoutDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup::<burn_wgpu::graphics::WebGpu>(&device, Default::default());
    device
}

#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
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
        assert!(matches!(
            BACKEND_NAME,
            "ndarray" | "metal" | "vulkan" | "webgpu"
        ));
    }
}
