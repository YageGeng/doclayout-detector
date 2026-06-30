use crate::error::LayoutError;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightInfo {
    pub name: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct PPDocLayoutV3Weights {
    bytes: Vec<u8>,
}

impl PPDocLayoutV3Weights {
    /// Reads safetensors bytes from disk and validates the model container.
    pub fn from_file(path: &Path) -> Result<Self, LayoutError> {
        let bytes = std::fs::read(path)
            .map_err(|error| LayoutError::InvalidModelOutput(format!("read weights: {error}")))?;
        Self::from_bytes(bytes)
    }

    /// Creates a weight store from in-memory safetensors bytes after validation.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, LayoutError> {
        let weights = Self { bytes };
        weights.validate()?;
        Ok(weights)
    }

    /// Returns all tensor names stored in the safetensors model file.
    pub fn names(&self) -> Result<Vec<String>, LayoutError> {
        let tensors = self.tensors()?;
        Ok(tensors.names().into_iter().map(str::to_string).collect())
    }

    /// Returns shape metadata for one named tensor without materializing it.
    pub fn info(&self, name: &str) -> Result<WeightInfo, LayoutError> {
        let tensors = self.tensors()?;
        let tensor = tensors.tensor(name).map_err(|error| {
            LayoutError::InvalidModelOutput(format!("load tensor {name}: {error}"))
        })?;
        Ok(WeightInfo {
            name: name.to_string(),
            shape: tensor.shape().to_vec(),
        })
    }

    /// Loads one F32 tensor with the expected rank onto the target backend device.
    pub fn tensor_f32<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, D>, LayoutError> {
        let (values, shape) = self.tensor_f32_values::<D>(name)?;
        Ok(Tensor::<B, D>::from_data(
            TensorData::new(values, shape),
            device,
        ))
    }

    /// Loads one F32 tensor into host values with the expected rank.
    pub(crate) fn tensor_f32_values<const D: usize>(
        &self,
        name: &str,
    ) -> Result<(Vec<f32>, [usize; D]), LayoutError> {
        let tensors = self.tensors()?;
        let tensor = tensors.tensor(name).map_err(|error| {
            LayoutError::InvalidModelOutput(format!("load tensor {name}: {error}"))
        })?;
        if tensor.dtype() != Dtype::F32 {
            return Err(LayoutError::InvalidModelOutput(format!(
                "expected tensor {name} dtype F32, got {:?}",
                tensor.dtype()
            )));
        }
        if tensor.shape().len() != D {
            return Err(LayoutError::InvalidModelOutput(format!(
                "expected tensor {name} rank {D}, got shape {:?}",
                tensor.shape()
            )));
        }

        let mut values = Vec::with_capacity(tensor.data().len() / 4);
        for chunk in tensor.data().chunks_exact(4) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let shape = tensor.shape().try_into().map_err(|_| {
            LayoutError::InvalidModelOutput(format!(
                "expected tensor {name} rank {D}, got shape {:?}",
                tensor.shape()
            ))
        })?;
        Ok((values, shape))
    }

    /// Validates that the safetensors payload can be deserialized.
    fn validate(&self) -> Result<(), LayoutError> {
        self.tensors().map(|_| ())
    }

    /// Deserializes the borrowed safetensors view used by all weight accessors.
    fn tensors(&self) -> Result<SafeTensors<'_>, LayoutError> {
        SafeTensors::deserialize(&self.bytes).map_err(|error| {
            LayoutError::InvalidModelOutput(format!("deserialize safetensors: {error}"))
        })
    }
}
