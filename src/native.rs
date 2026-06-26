use crate::PageImage;
use crate::annotate::{AnnotatedDetection, annotate_page_rgba};
use crate::model::EmbeddedModel;
use crate::pp_doclayout::{PPDocLayoutV3Detector, PPDocLayoutV3Options};
use pdfium::Library;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::Level;

const DEFAULT_DPI: f32 = 96.0;

#[derive(Debug, Clone, PartialEq)]
struct NativeCliArgs {
    input_pdf: PathBuf,
    output_dir: PathBuf,
    dpi: f32,
}

impl NativeCliArgs {
    /// Parse CLI arguments and return `None` when usage should be shown.
    fn from_args(args: &[String]) -> Result<Option<Self>, String> {
        let Some(input_pdf) = args.get(1) else {
            return Ok(None);
        };
        let Some(output_dir) = args.get(2) else {
            return Ok(None);
        };
        let dpi = match args.get(3) {
            Some(value) => value
                .parse::<f32>()
                .map_err(|error| format!("invalid dpi '{value}': {error}"))?,
            None => DEFAULT_DPI,
        };

        Ok(Some(Self {
            input_pdf: PathBuf::from(input_pdf),
            output_dir: PathBuf::from(output_dir),
            dpi,
        }))
    }
}

/// Run the native PDF layout detection CLI.
pub fn run_cli() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    let Some(cli_args) = NativeCliArgs::from_args(&args)? else {
        print_usage(&args[0]);
        return Ok(());
    };

    detect_pdf_to_dir(&cli_args.input_pdf, &cli_args.output_dir, cli_args.dpi)
}

/// Detect every page in a PDF, write annotated outputs, and emit per-stage timing events.
pub fn detect_pdf_to_dir(input_pdf: &Path, output_dir: &Path, dpi: f32) -> Result<(), String> {
    let total_started = Instant::now();

    let create_output_started = Instant::now();
    fs::create_dir_all(output_dir)
        .map_err(|error| format!("create output dir {}: {error}", output_dir.display()))?;
    tracing::event!(
        Level::INFO,
        step = "create_output_dir",
        duration_ms = create_output_started.elapsed().as_secs_f64() * 1000.0,
        output_dir = %output_dir.display(),
        "native pipeline step completed"
    );

    let model_started = Instant::now();
    let model = EmbeddedModel::new().map_err(|error| format!("initialize model: {error}"))?;
    let detector = PPDocLayoutV3Detector::new(model, PPDocLayoutV3Options::default());
    tracing::event!(
        Level::INFO,
        step = "initialize_model",
        duration_ms = model_started.elapsed().as_secs_f64() * 1000.0,
        "native pipeline step completed"
    );

    let read_pdf_started = Instant::now();
    let pdf_bytes = fs::read(input_pdf)
        .map_err(|error| format!("read PDF {}: {error}", input_pdf.display()))?;
    tracing::event!(
        Level::INFO,
        step = "read_pdf",
        duration_ms = read_pdf_started.elapsed().as_secs_f64() * 1000.0,
        input_pdf = %input_pdf.display(),
        bytes = pdf_bytes.len(),
        "native pipeline step completed"
    );

    let load_pdf_started = Instant::now();
    let lib = Library::init();
    let document = lib
        .load_document_from_bytes(&pdf_bytes, None)
        .map_err(|error| format!("load PDF {}: {error}", input_pdf.display()))?;
    let page_count = document.page_count();
    tracing::event!(
        Level::INFO,
        step = "load_pdf",
        duration_ms = load_pdf_started.elapsed().as_secs_f64() * 1000.0,
        pages = page_count,
        "native pipeline step completed"
    );

    for page_index in 0..page_count {
        let page_started = Instant::now();
        let page_number = page_index as u32 + 1;

        let load_page_started = Instant::now();
        let page = document
            .page(page_index)
            .map_err(|error| format!("load page {page_number}: {error}"))?;
        let page_width = page.width();
        let page_height = page.height();
        let load_page_ms = load_page_started.elapsed().as_secs_f64() * 1000.0;

        let render_started = Instant::now();
        let bitmap = page
            .render(dpi)
            .map_err(|error| format!("render page {page_number}: {error}"))?;
        let width = bitmap.width() as u32;
        let height = bitmap.height() as u32;
        let render_ms = render_started.elapsed().as_secs_f64() * 1000.0;

        let bitmap_started = Instant::now();
        let rgb = bitmap.to_rgb();
        let mut rgba = bitmap.to_rgba();
        let bitmap_ms = bitmap_started.elapsed().as_secs_f64() * 1000.0;

        let image = PageImage {
            rgb: &rgb,
            width,
            height,
            page_width,
            page_height,
            dpi,
        };
        let detect_started = Instant::now();
        let detections = detector
            .detect_page(&image)
            .map_err(|error| format!("detect page {page_number}: {error}"))?;
        let detect_ms = detect_started.elapsed().as_secs_f64() * 1000.0;

        let annotate_started = Instant::now();
        let annotated = detections
            .iter()
            .map(AnnotatedDetection::from)
            .collect::<Vec<_>>();

        annotate_page_rgba(
            &mut rgba,
            width,
            height,
            page_width,
            page_height,
            &annotated,
        );
        let annotate_ms = annotate_started.elapsed().as_secs_f64() * 1000.0;

        let encode_png_started = Instant::now();
        let png_bytes = encode_png_rgba(&rgba, width, height)
            .map_err(|error| format!("encode page {page_number} PNG: {error}"))?;
        let encode_png_ms = encode_png_started.elapsed().as_secs_f64() * 1000.0;

        let image_path = output_path_for_page(output_dir, page_number, "png");
        let json_path = output_path_for_page(output_dir, page_number, "json");

        let write_png_started = Instant::now();
        fs::write(&image_path, png_bytes)
            .map_err(|error| format!("write {}: {error}", image_path.display()))?;
        let write_png_ms = write_png_started.elapsed().as_secs_f64() * 1000.0;

        let serialize_json_started = Instant::now();
        let json = NativePageOutput {
            page_number,
            width,
            height,
            page_width,
            page_height,
            dpi,
            detections: &annotated,
        };
        let json = serde_json::to_vec_pretty(&json)
            .map_err(|error| format!("serialize page {page_number} JSON: {error}"))?;
        let serialize_json_ms = serialize_json_started.elapsed().as_secs_f64() * 1000.0;

        let write_json_started = Instant::now();
        fs::write(&json_path, json)
            .map_err(|error| format!("write {}: {error}", json_path.display()))?;
        let write_json_ms = write_json_started.elapsed().as_secs_f64() * 1000.0;

        let total_page_ms = page_started.elapsed().as_secs_f64() * 1000.0;

        tracing::event!(
            Level::INFO,
            page_number,
            page_count,
            dpi,
            width,
            height,
            page_width,
            page_height,
            image = %image_path.display(),
            json = %json_path.display(),
            boxes = annotated.len(),
            load_page_ms,
            render_ms,
            bitmap_ms,
            detect_ms,
            annotate_ms,
            encode_png_ms,
            write_png_ms,
            serialize_json_ms,
            write_json_ms,
            total_page_ms,
            "native page completed"
        );
    }

    tracing::event!(
        Level::INFO,
        input_pdf = %input_pdf.display(),
        output_dir = %output_dir.display(),
        dpi,
        pages = page_count,
        total_ms = total_started.elapsed().as_secs_f64() * 1000.0,
        "native pipeline completed"
    );

    Ok(())
}

/// Print the CLI usage as a tracing event so logging remains structured.
fn print_usage(bin: &str) {
    tracing::event!(
        Level::INFO,
        usage = %format!("{bin} <input.pdf> <output-dir> [dpi]"),
        example = %format!("cargo run --no-default-features --features backend-ndarray,native-cli -- file.pdf out 96"),
        "native CLI usage"
    );
}

/// Encode an RGBA page buffer as PNG bytes.
fn encode_png_rgba(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, png::EncodingError> {
    let mut png_bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut png_bytes, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(rgba)?;
    }
    Ok(png_bytes)
}

/// Build a stable output file path for one-based page numbers.
fn output_path_for_page(output_dir: &Path, page_number: u32, extension: &str) -> PathBuf {
    output_dir.join(format!("page-{page_number:04}.{extension}"))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NativePageOutput<'a> {
    page_number: u32,
    width: u32,
    height: u32,
    page_width: f32,
    page_height: f32,
    dpi: f32,
    detections: &'a [AnnotatedDetection],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_cli_args_use_96_dpi_when_dpi_is_omitted() {
        let args = vec![
            "doclayout-detector".to_string(),
            "input.pdf".to_string(),
            "out".to_string(),
        ];

        let parsed = NativeCliArgs::from_args(&args).unwrap().unwrap();

        assert_eq!(parsed.dpi, 96.0);
    }

    #[test]
    fn native_cli_args_use_explicit_dpi_when_present() {
        let args = vec![
            "doclayout-detector".to_string(),
            "input.pdf".to_string(),
            "out".to_string(),
            "150".to_string(),
        ];

        let parsed = NativeCliArgs::from_args(&args).unwrap().unwrap();

        assert_eq!(parsed.dpi, 150.0);
    }

    #[test]
    fn output_path_uses_padded_one_based_page_number() {
        assert_eq!(
            output_path_for_page(Path::new("out"), 7, "png"),
            Path::new("out/page-0007.png")
        );
    }
}
