mod annotate;
pub mod error;
pub mod model;
#[cfg(all(not(target_arch = "wasm32"), feature = "native-cli"))]
pub mod native;
pub mod pp_doclayout;
pub mod preprocess;
pub mod types;
#[cfg(all(target_arch = "wasm32", feature = "backend-webgpu"))]
mod wasi_stubs;
#[cfg(all(target_arch = "wasm32", feature = "backend-webgpu"))]
mod wasm;

pub use annotate::{AnnotatedDetection, annotate_page_rgba};
pub use error::LayoutError;
pub use types::{LayoutDetection, PageImage};
