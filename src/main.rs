#[cfg(feature = "cli")]
mod cli;

/// Starts the native CLI on desktop targets and stays inert for wasm library builds.
fn main() {
    #[cfg(feature = "cli")]
    cli::run();
}
