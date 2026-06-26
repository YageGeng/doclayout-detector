use crate::PageImage;
use crate::annotate::{AnnotatedDetection, annotate_page_rgba};
use crate::model::EmbeddedModel;
use crate::pp_doclayout::{PPDocLayoutV3Detector, PPDocLayoutV3Options};
use pdfium::{Bitmap, Library};
use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
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
    pub fn new() -> Self {
        Self {
            options: PPDocLayoutV3Options::default(),
            detector: RefCell::new(None),
            pdf_data: RefCell::new(None),
            page_count: Cell::new(0),
        }
    }

    #[wasm_bindgen(js_name = loadPdf)]
    pub fn load_pdf(&self, data: Vec<u8>) -> Result<u32, JsError> {
        let page_count = pdf_page_count(&data)?;
        *self.pdf_data.borrow_mut() = Some(data);
        self.page_count.set(page_count);
        tracing::info!(pages = page_count, "loaded PDF into wasm cache");
        Ok(page_count)
    }

    #[wasm_bindgen(js_name = loadedPageCount)]
    pub fn loaded_page_count(&self) -> u32 {
        self.page_count.get()
    }

    #[wasm_bindgen(js_name = pageCount)]
    pub fn page_count(&self, data: Vec<u8>) -> Result<u32, JsError> {
        pdf_page_count(&data)
    }

    #[wasm_bindgen(js_name = detectPage)]
    pub async fn detect_page(
        &self,
        data: Vec<u8>,
        page_number: u32,
        dpi: f32,
    ) -> Result<JsValue, JsError> {
        let detector = self.detector().await?;
        let rendered = render_pdf_page(&data, page_number, dpi)?;
        self.detect_rendered_page(detector, rendered, page_number, dpi)
            .await
    }

    #[wasm_bindgen(js_name = detectLoadedPage)]
    pub async fn detect_loaded_page(&self, page_number: u32, dpi: f32) -> Result<JsValue, JsError> {
        let detector = self.detector().await?;
        let rendered = {
            let pdf_data = self.pdf_data.borrow();
            let data = pdf_data
                .as_deref()
                .ok_or_else(|| JsError::new("no PDF loaded; call loadPdf first"))?;
            render_pdf_page(data, page_number, dpi)?
        };
        self.detect_rendered_page(detector, rendered, page_number, dpi)
            .await
    }
}

impl PPDocLayoutWasm {
    async fn detect_rendered_page(
        &self,
        detector: PPDocLayoutV3Detector<EmbeddedModel>,
        rendered: RenderedPage,
        page_number: u32,
        dpi: f32,
    ) -> Result<JsValue, JsError> {
        let image = PageImage {
            rgb: &rendered.rgb,
            width: rendered.width,
            height: rendered.height,
            page_width: rendered.page_width,
            page_height: rendered.page_height,
            dpi,
        };
        let detections = detector
            .detect_page_async(&image)
            .await
            .map_err(|error| JsError::new(&format!("layout detection failed: {error}")))?;

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
        let png_bytes = encode_png_rgba(&rgba, rendered.width, rendered.height)?;

        let result = WasmPageResult {
            page_number,
            width: rendered.width,
            height: rendered.height,
            page_width: rendered.page_width,
            page_height: rendered.page_height,
            detections: &annotated,
            image_bytes: &png_bytes,
        };

        serde_wasm_bindgen::to_value(&result)
            .map_err(|error| JsError::new(&format!("failed to encode page result: {error}")))
    }

    async fn detector(&self) -> Result<PPDocLayoutV3Detector<EmbeddedModel>, JsError> {
        if let Some(detector) = self.detector.borrow().clone() {
            return Ok(detector);
        }

        let model = EmbeddedModel::new_async()
            .await
            .map_err(|error| JsError::new(&format!("failed to initialize model: {error}")))?;
        let detector = PPDocLayoutV3Detector::new(model, self.options.clone());
        *self.detector.borrow_mut() = Some(detector.clone());
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

fn pdf_page_count(data: &[u8]) -> Result<u32, JsError> {
    let lib = Library::init();
    let document = lib
        .load_document_from_bytes(data, None)
        .map_err(|error| JsError::new(&format!("failed to load PDF: {error}")))?;
    Ok(document.page_count() as u32)
}

fn render_pdf_page(data: &[u8], page_number: u32, dpi: f32) -> Result<RenderedPage, JsError> {
    if page_number == 0 {
        return Err(JsError::new(
            "pageNumber is 1-based and must be greater than 0",
        ));
    }

    let lib = Library::init();
    let document = lib
        .load_document_from_bytes(data, None)
        .map_err(|error| JsError::new(&format!("failed to load PDF: {error}")))?;
    let page_count = document.page_count() as u32;
    if page_number > page_count {
        return Err(JsError::new(&format!(
            "page {page_number} out of range; document has {page_count} pages"
        )));
    }

    let page = document
        .page((page_number - 1) as i32)
        .map_err(|error| JsError::new(&format!("failed to load page {page_number}: {error}")))?;
    let page_width = page.width();
    let page_height = page.height();
    let bitmap = page
        .render(dpi)
        .map_err(|error| JsError::new(&format!("failed to render page {page_number}: {error}")))?;

    let (rgb, rgba) = bitmap_to_rgb_and_rgba(&bitmap);

    Ok(RenderedPage {
        width: bitmap.width() as u32,
        height: bitmap.height() as u32,
        page_width,
        page_height,
        rgb,
        rgba,
    })
}

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
