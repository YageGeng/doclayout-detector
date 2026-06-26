//! Stub implementations of libc functions that pdfium's statically-linked
//! wasi-libc expects at runtime under the "env" module namespace.
//!
//! WASI preview1 syscalls (wasi_snapshot_preview1::*) cannot be stubbed from
//! Rust because they live in a different WASM import module namespace. Those
//! are provided in JavaScript — see packages/wasm/scripts/patch-wasi-imports.js.

// ---------------------------------------------------------------------------
// env:: stubs (libc / pthreads)
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn getpid() -> i32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn pthread_mutex_init(_mutex: *mut u8, _attr: *const u8) -> i32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn pthread_mutex_lock(_mutex: *mut u8) -> i32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn pthread_mutex_unlock(_mutex: *mut u8) -> i32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn pthread_mutex_destroy(_mutex: *mut u8) -> i32 {
    0
}
