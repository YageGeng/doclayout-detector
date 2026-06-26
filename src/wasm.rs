use crate::PageImage;
use crate::annotate::{AnnotatedDetection, annotate_page_rgba};
use crate::model::EmbeddedModel;
use crate::pp_doclayout::{PPDocLayoutV3Detector, PPDocLayoutV3Options};
use pdfium::{Bitmap, Library};
use std::cell::{Cell, RefCell};
use tracing::Level;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
/// Initializes browser-side panic reporting and tracing before exported APIs run.
pub fn wasm_start() {
    #[cfg(feature = "panic_hook")]
    console_error_panic_hook::set_once();
    tracing_wasm::set_as_global_default();
}

#[wasm_bindgen]
pub struct PPDocLayoutWasm {
    options: PPDocLayoutV3Options,
    detector: RefCell<Option<PPDocLayoutV3Detector<EmbeddedModel>>>,
    pdf_data: RefCell<Option<Vec<u8>>>,
    page_count: Cell<u32>,
}

#[wasm_bindgen]
impl PPDocLayoutWasm {
    #[wasm_bindgen(constructor)]
    /// Creates a wasm facade with lazy model initialization and an empty PDF cache.
    pub fn new() -> Self {
        Self {
            options: PPDocLayoutV3Options::default(),
            detector: RefCell::new(None),
            pdf_data: RefCell::new(None),
            page_count: Cell::new(0),
        }
    }

    #[wasm_bindgen(js_name = loadPdf)]
    /// Stores PDF bytes in the wasm instance and returns the document page count.
    pub fn load_pdf(&self, data: Vec<u8>) -> Result<u32, JsError> {
        let started = WasmTimer::start();
        let page_count = pdf_page_count(&data)?;
        *self.pdf_data.borrow_mut() = Some(data);
        self.page_count.set(page_count);
        tracing::event!(
            Level::INFO,
            pages = page_count,
            bytes = self.pdf_data.borrow().as_ref().map_or(0, Vec::len),
            duration_ms = started.elapsed_ms(),
            "loaded PDF into wasm cache"
        );
        Ok(page_count)
    }

    #[wasm_bindgen(js_name = loadedPageCount)]
    /// Returns the page count for the PDF currently cached by `loadPdf`.
    pub fn loaded_page_count(&self) -> u32 {
        self.page_count.get()
    }

    #[wasm_bindgen(js_name = pageCount)]
    /// Parses PDF bytes and returns the page count without mutating the cached document.
    pub fn page_count(&self, data: Vec<u8>) -> Result<u32, JsError> {
        pdf_page_count(&data)
    }

    #[wasm_bindgen(js_name = detectPage)]
    /// Renders and detects a page from the provided PDF bytes in one call.
    pub async fn detect_page(
        &self,
        data: Vec<u8>,
        page_number: u32,
        dpi: f32,
    ) -> Result<JsValue, JsError> {
        let detector = self.detector().await?;
        let render_started = WasmTimer::start();
        let rendered = render_pdf_page(&data, page_number, dpi)?;
        tracing::event!(
            Level::INFO,
            page_number,
            dpi,
            duration_ms = render_started.elapsed_ms(),
            width = rendered.width,
            height = rendered.height,
            "wasm page render completed"
        );
        self.detect_rendered_page(detector, rendered, page_number, dpi)
            .await
    }

    #[wasm_bindgen(js_name = detectLoadedPage)]
    /// Renders and detects a page from the PDF previously stored by `loadPdf`.
    pub async fn detect_loaded_page(&self, page_number: u32, dpi: f32) -> Result<JsValue, JsError> {
        let detector = self.detector().await?;
        let render_started = WasmTimer::start();
        let rendered = {
            let pdf_data = self.pdf_data.borrow();
            let data = pdf_data
                .as_deref()
                .ok_or_else(|| JsError::new("no PDF loaded; call loadPdf first"))?;
            render_pdf_page(data, page_number, dpi)?
        };
        tracing::event!(
            Level::INFO,
            page_number,
            dpi,
            duration_ms = render_started.elapsed_ms(),
            width = rendered.width,
            height = rendered.height,
            "wasm page render completed"
        );
        self.detect_rendered_page(detector, rendered, page_number, dpi)
            .await
    }

    #[wasm_bindgen(js_name = detectLoadedPages)]
    /// Renders and detects a contiguous batch from the PDF previously stored by `loadPdf`.
    pub async fn detect_loaded_pages(
        &self,
        start_page: u32,
        count: u32,
        dpi: f32,
    ) -> Result<JsValue, JsError> {
        let detector = self.detector().await?;
        let render_started = WasmTimer::start();
        let rendered_pages = {
            let pdf_data = self.pdf_data.borrow();
            let data = pdf_data
                .as_deref()
                .ok_or_else(|| JsError::new("no PDF loaded; call loadPdf first"))?;
            render_pdf_pages(data, start_page, count, dpi)?
        };
        tracing::event!(
            Level::INFO,
            start_page,
            count,
            dpi,
            duration_ms = render_started.elapsed_ms(),
            "wasm page batch render completed"
        );
        self.detect_rendered_pages(detector, rendered_pages, dpi)
            .await
    }
}

impl PPDocLayoutWasm {
    /// Runs detection, annotation, PNG encoding, and JS serialization for a rendered page.
    async fn detect_rendered_page(
        &self,
        detector: PPDocLayoutV3Detector<EmbeddedModel>,
        rendered: RenderedPage,
        page_number: u32,
        dpi: f32,
    ) -> Result<JsValue, JsError> {
        let total_started = WasmTimer::start();
        let image = PageImage {
            rgb: &rendered.rgb,
            width: rendered.width,
            height: rendered.height,
            page_width: rendered.page_width,
            page_height: rendered.page_height,
            dpi,
        };
        let detect_started = WasmTimer::start();
        let detections = detector
            .detect_page_async(&image)
            .await
            .map_err(|error| JsError::new(&format!("layout detection failed: {error}")))?;
        let detect_ms = detect_started.elapsed_ms();

        let annotate_started = WasmTimer::start();
        let annotated = detections
            .iter()
            .map(AnnotatedDetection::from)
            .collect::<Vec<_>>();
        let mut rgba = rendered.rgba;
        annotate_page_rgba(
            &mut rgba,
            rendered.width,
            rendered.height,
            rendered.page_width,
            rendered.page_height,
            &annotated,
        );
        let annotate_ms = annotate_started.elapsed_ms();

        let encode_started = WasmTimer::start();
        let png_bytes = encode_png_rgba(&rgba, rendered.width, rendered.height)?;
        let encode_png_ms = encode_started.elapsed_ms();

        let result = WasmPageResult {
            page_number,
            width: rendered.width,
            height: rendered.height,
            page_width: rendered.page_width,
            page_height: rendered.page_height,
            detections: &annotated,
            image_bytes: &png_bytes,
        };

        let serialize_started = WasmTimer::start();
        let value = serde_wasm_bindgen::to_value(&result)
            .map_err(|error| JsError::new(&format!("failed to encode page result: {error}")))?;
        let serialize_ms = serialize_started.elapsed_ms();

        tracing::event!(
            Level::INFO,
            page_number,
            dpi,
            width = rendered.width,
            height = rendered.height,
            detections = annotated.len(),
            image_bytes = png_bytes.len(),
            detect_ms,
            annotate_ms,
            encode_png_ms,
            serialize_ms,
            total_ms = total_started.elapsed_ms(),
            "wasm page pipeline completed"
        );

        Ok(value)
    }

    /// Runs batched detection, annotation, PNG encoding, and JS serialization for rendered pages.
    async fn detect_rendered_pages(
        &self,
        detector: PPDocLayoutV3Detector<EmbeddedModel>,
        mut rendered_pages: Vec<(u32, RenderedPage)>,
        dpi: f32,
    ) -> Result<JsValue, JsError> {
        let total_started = WasmTimer::start();
        let images = rendered_pages
            .iter()
            .map(|(_, rendered)| rendered.image(dpi))
            .collect::<Vec<_>>();

        let detect_started = WasmTimer::start();
        let detections_by_page = detector
            .detect_pages_async(&images)
            .await
            .map_err(|error| JsError::new(&format!("layout batch detection failed: {error}")))?;
        let detect_ms = detect_started.elapsed_ms();
        drop(images);

        let mut results = Vec::with_capacity(rendered_pages.len());
        for ((page_number, rendered), detections) in
            rendered_pages.drain(..).zip(detections_by_page)
        {
            let annotate_started = WasmTimer::start();
            let annotated = detections
                .iter()
                .map(AnnotatedDetection::from)
                .collect::<Vec<_>>();
            let mut rgba = rendered.rgba;
            annotate_page_rgba(
                &mut rgba,
                rendered.width,
                rendered.height,
                rendered.page_width,
                rendered.page_height,
                &annotated,
            );
            let annotate_ms = annotate_started.elapsed_ms();

            let encode_started = WasmTimer::start();
            let png_bytes = encode_png_rgba(&rgba, rendered.width, rendered.height)?;
            let encode_png_ms = encode_started.elapsed_ms();

            tracing::event!(
                Level::INFO,
                page_number,
                dpi,
                width = rendered.width,
                height = rendered.height,
                detections = annotated.len(),
                image_bytes = png_bytes.len(),
                detect_ms,
                annotate_ms,
                encode_png_ms,
                "wasm page batch item completed"
            );

            results.push(WasmOwnedPageResult {
                page_number,
                width: rendered.width,
                height: rendered.height,
                page_width: rendered.page_width,
                page_height: rendered.page_height,
                detections: annotated,
                image_bytes: png_bytes,
            });
        }

        let serialize_started = WasmTimer::start();
        let value = serde_wasm_bindgen::to_value(&results)
            .map_err(|error| JsError::new(&format!("failed to encode batch result: {error}")))?;
        let serialize_ms = serialize_started.elapsed_ms();

        tracing::event!(
            Level::INFO,
            pages = results.len(),
            detect_ms,
            serialize_ms,
            total_ms = total_started.elapsed_ms(),
            "wasm page batch pipeline completed"
        );

        Ok(value)
    }

    /// Lazily initializes and caches the layout detector used by wasm entrypoints.
    async fn detector(&self) -> Result<PPDocLayoutV3Detector<EmbeddedModel>, JsError> {
        if let Some(detector) = self.detector.borrow().clone() {
            return Ok(detector);
        }

        let started = WasmTimer::start();
        let model = EmbeddedModel::new_async()
            .await
            .map_err(|error| JsError::new(&format!("failed to initialize model: {error}")))?;
        let detector = PPDocLayoutV3Detector::new(model, self.options.clone());
        *self.detector.borrow_mut() = Some(detector.clone());
        tracing::event!(
            Level::INFO,
            duration_ms = started.elapsed_ms(),
            "wasm detector initialized"
        );
        Ok(detector)
    }
}

struct RenderedPage {
    width: u32,
    height: u32,
    page_width: f32,
    page_height: f32,
    rgb: Vec<u8>,
    rgba: Vec<u8>,
}

impl RenderedPage {
    /// Borrows this rendered page as model input without copying pixel buffers.
    fn image(&self, dpi: f32) -> PageImage<'_> {
        PageImage {
            rgb: &self.rgb,
            width: self.width,
            height: self.height,
            page_width: self.page_width,
            page_height: self.page_height,
            dpi,
        }
    }
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct WasmPageResult<'a> {
    page_number: u32,
    width: u32,
    height: u32,
    page_width: f32,
    page_height: f32,
    detections: &'a [AnnotatedDetection],
    #[serde(with = "serde_bytes")]
    image_bytes: &'a [u8],
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct WasmOwnedPageResult {
    page_number: u32,
    width: u32,
    height: u32,
    page_width: f32,
    page_height: f32,
    detections: Vec<AnnotatedDetection>,
    #[serde(with = "serde_bytes")]
    image_bytes: Vec<u8>,
}

/// Counts pages in a PDF byte buffer using Pdfium.
fn pdf_page_count(data: &[u8]) -> Result<u32, JsError> {
    let lib = Library::init();
    let document = lib
        .load_document_from_bytes(data, None)
        .map_err(|error| JsError::new(&format!("failed to load PDF: {error}")))?;
    Ok(document.page_count() as u32)
}

/// Renders a one-based PDF page into RGB input bytes and RGBA display bytes.
fn render_pdf_page(data: &[u8], page_number: u32, dpi: f32) -> Result<RenderedPage, JsError> {
    let mut pages = render_pdf_pages(data, page_number, 1, dpi)?;
    Ok(pages.remove(0).1)
}

/// Renders a contiguous one-based PDF page range into RGB/RGBA page buffers.
fn render_pdf_pages(
    data: &[u8],
    start_page: u32,
    count: u32,
    dpi: f32,
) -> Result<Vec<(u32, RenderedPage)>, JsError> {
    if start_page == 0 {
        return Err(JsError::new(
            "startPage is 1-based and must be greater than 0",
        ));
    }
    if count == 0 {
        return Err(JsError::new("count must be greater than 0"));
    }
    let end_page = start_page
        .checked_add(count - 1)
        .ok_or_else(|| JsError::new("page range overflow"))?;

    let load_started = WasmTimer::start();
    let lib = Library::init();
    let document = lib
        .load_document_from_bytes(data, None)
        .map_err(|error| JsError::new(&format!("failed to load PDF: {error}")))?;
    let page_count = document.page_count() as u32;
    tracing::event!(
        Level::INFO,
        start_page,
        end_page,
        page_count,
        duration_ms = load_started.elapsed_ms(),
        "wasm PDF document loaded for page batch render"
    );
    if end_page > page_count {
        return Err(JsError::new(&format!(
            "page range {start_page}-{end_page} out of range; document has {page_count} pages"
        )));
    }

    let mut pages = Vec::with_capacity(count as usize);
    for page_number in start_page..=end_page {
        let page_started = WasmTimer::start();
        let page = document.page((page_number - 1) as i32).map_err(|error| {
            JsError::new(&format!("failed to load page {page_number}: {error}"))
        })?;
        let page_width = page.width();
        let page_height = page.height();
        tracing::event!(
            Level::INFO,
            page_number,
            page_width,
            page_height,
            duration_ms = page_started.elapsed_ms(),
            "wasm PDF page loaded"
        );

        let raster_started = WasmTimer::start();
        let bitmap = page.render(dpi).map_err(|error| {
            JsError::new(&format!("failed to render page {page_number}: {error}"))
        })?;
        tracing::event!(
            Level::INFO,
            page_number,
            dpi,
            width = bitmap.width(),
            height = bitmap.height(),
            duration_ms = raster_started.elapsed_ms(),
            "wasm PDF page rasterized"
        );

        let bitmap_started = WasmTimer::start();
        let (rgb, rgba) = bitmap_to_rgb_and_rgba(&bitmap);
        tracing::event!(
            Level::INFO,
            page_number,
            rgb_bytes = rgb.len(),
            rgba_bytes = rgba.len(),
            duration_ms = bitmap_started.elapsed_ms(),
            "wasm bitmap converted"
        );

        pages.push((
            page_number,
            RenderedPage {
                width: bitmap.width() as u32,
                height: bitmap.height() as u32,
                page_width,
                page_height,
                rgb,
                rgba,
            },
        ));
    }

    Ok(pages)
}

/// Converts Pdfium BGRA bitmap rows into RGB model input and RGBA annotation buffers.
fn bitmap_to_rgb_and_rgba(bitmap: &Bitmap<'_>) -> (Vec<u8>, Vec<u8>) {
    let width = bitmap.width() as usize;
    let height = bitmap.height() as usize;
    let stride = bitmap.stride() as usize;
    let src = bitmap.buffer();
    let mut rgb = Vec::with_capacity(width * height * 3);
    let mut rgba = Vec::with_capacity(width * height * 4);

    for y in 0..height {
        let row = &src[y * stride..y * stride + width * 4];
        for pixel in row.chunks_exact(4) {
            let red = pixel[2];
            let green = pixel[1];
            let blue = pixel[0];
            let alpha = pixel[3];
            rgb.extend_from_slice(&[red, green, blue]);
            rgba.extend_from_slice(&[red, green, blue, alpha]);
        }
    }

    (rgb, rgba)
}

/// Encodes an RGBA page buffer as PNG bytes for transfer to JavaScript.
fn encode_png_rgba(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsError> {
    let mut png_bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut png_bytes, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|error| JsError::new(&format!("failed to create PNG: {error}")))?;
        writer
            .write_image_data(rgba)
            .map_err(|error| JsError::new(&format!("failed to write PNG: {error}")))?;
    }
    Ok(png_bytes)
}

#[derive(Debug, Clone)]
struct WasmTimer {
    started_ms: f64,
}

impl WasmTimer {
    /// Start a browser-compatible timer for wasm tracing events.
    fn start() -> Self {
        Self {
            started_ms: js_sys::Date::now(),
        }
    }

    /// Return elapsed milliseconds using JavaScript's clock.
    fn elapsed_ms(&self) -> f64 {
        js_sys::Date::now() - self.started_ms
    }
}
