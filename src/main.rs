#[cfg(not(target_arch = "wasm32"))]
mod cli;

/// Starts the native CLI on desktop targets and stays inert for wasm library builds.
fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    cli::run();
}
