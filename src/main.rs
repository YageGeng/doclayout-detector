#[cfg(all(not(target_arch = "wasm32"), feature = "native-cli"))]
fn main() {
    init_tracing();
    if let Err(error) = doclayout_detector::native::run_cli() {
        tracing::error!("{error}");
        std::process::exit(1);
    }
}

#[cfg(all(not(target_arch = "wasm32"), not(feature = "native-cli")))]
fn main() {
    init_tracing();
    tracing::error!(
        "native CLI is disabled; rebuild with --features native-cli plus one backend feature"
    );
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn init_tracing() {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}
