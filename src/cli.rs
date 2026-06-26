use clap::{Args, Parser, Subcommand};
use pdfium::Library;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::Level;

use doclayout_detector::model::EmbeddedModel;
use doclayout_detector::pp_doclayout::{PPDocLayoutV3Detector, PPDocLayoutV3Options};
use doclayout_detector::{AnnotatedDetection, PageImage, annotate_page_rgba};

const DEFAULT_DPI: f32 = 96.0;

#[derive(Debug, Clone, PartialEq, Parser)]
#[command(author, version, about = "Detect document layout boxes in PDF pages.")]
struct NativeCli {
    #[command(subcommand)]
    command: NativeCommand,
}

#[derive(Debug, Clone, PartialEq, Subcommand)]
enum NativeCommand {
    Detect(DetectArgs),
}

#[derive(Debug, Clone, PartialEq, Args)]
struct DetectArgs {
    input_pdf: PathBuf,
    #[arg(short = 'o', long = "output-dir", default_value = "output")]
    output_dir: PathBuf,
    #[arg(long, default_value_t = DEFAULT_DPI)]
    dpi: f32,
    #[arg(long = "batch-size", default_value_t = 1, value_parser = BatchSize::parse)]
    batch_size: usize,
}

impl NativeCli {
    /// Parses command-line arguments using clap's enum-based subcommand model.
    fn from_env() -> Self {
        Self::parse()
    }

    /// Dispatches the parsed native command.
    fn run(self) -> Result<(), String> {
        self.command.run()
    }
}

impl NativeCommand {
    /// Runs the selected native CLI command.
    fn run(self) -> Result<(), String> {
        match self {
            Self::Detect(args) => args.run(),
        }
    }
}

impl DetectArgs {
    /// Runs PDF detection using the parsed detect command options.
    fn run(self) -> Result<(), String> {
        detect_pdf_to_dir(&self.input_pdf, &self.output_dir, self.dpi, self.batch_size)
    }
}

struct BatchSize;

impl BatchSize {
    /// Parses a batch size constrained to the memory-safe supported range.
    fn parse(value: &str) -> Result<usize, String> {
        let parsed = value
            .parse::<usize>()
            .map_err(|error| format!("batch size must be an integer: {error}"))?;
        if !(1..=4).contains(&parsed) {
            return Err("batch size must be between 1 and 4".to_string());
        }
        Ok(parsed)
    }
}

/// Runs the native CLI process and exits with status 1 on pipeline errors.
pub fn run() {
    init_tracing();
    if let Err(error) = NativeCli::from_env().run() {
        tracing::error!("{error}");
        std::process::exit(1);
    }
}

/// Detects every page in a PDF, writes annotated outputs, and emits per-stage timing events.
fn detect_pdf_to_dir(
    input_pdf: &Path,
    output_dir: &Path,
    dpi: f32,
    batch_size: usize,
) -> Result<(), String> {
    if batch_size == 0 {
        return Err("batch size must be greater than zero".to_string());
    }

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
    let page_count = usize::try_from(document.page_count())
        .map_err(|error| format!("convert PDF page count: {error}"))?;
    tracing::event!(
        Level::INFO,
        step = "load_pdf",
        duration_ms = load_pdf_started.elapsed().as_secs_f64() * 1000.0,
        pages = page_count,
        "native pipeline step completed"
    );

    let mut batch_start = 0usize;
    while batch_start < page_count {
        let batch_end = (batch_start + batch_size).min(page_count);
        let batch_started = Instant::now();
        let mut rendered_pages = Vec::with_capacity(batch_end - batch_start);

        for page_index in batch_start..batch_end {
            let page_started = Instant::now();
            let page_number = page_index as u32 + 1;
            let pdf_page_index = i32::try_from(page_index)
                .map_err(|error| format!("convert page index: {error}"))?;

            let load_page_started = Instant::now();
            let page = document
                .page(pdf_page_index)
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
            let rgba = bitmap.to_rgba();
            let bitmap_ms = bitmap_started.elapsed().as_secs_f64() * 1000.0;

            rendered_pages.push(RenderedPage {
                page_started,
                page_number,
                width,
                height,
                page_width,
                page_height,
                rgb,
                rgba,
                load_page_ms,
                render_ms,
                bitmap_ms,
            });
        }

        let images = rendered_pages
            .iter()
            .map(|page| page.image(dpi))
            .collect::<Vec<_>>();
        let detect_started = Instant::now();
        let detections_by_page = detector
            .detect_pages(&images)
            .map_err(|error| format!("detect pages {}-{}: {error}", batch_start + 1, batch_end))?;
        let batch_detect_ms = detect_started.elapsed().as_secs_f64() * 1000.0;
        drop(images);

        tracing::event!(
            Level::INFO,
            first_page = batch_start + 1,
            last_page = batch_end,
            pages = rendered_pages.len(),
            batch_detect_ms,
            total_batch_ms = batch_started.elapsed().as_secs_f64() * 1000.0,
            "native batch completed"
        );

        for (rendered, detections) in rendered_pages.iter_mut().zip(detections_by_page) {
            let annotate_started = Instant::now();
            let annotated = detections
                .iter()
                .map(AnnotatedDetection::from)
                .collect::<Vec<_>>();

            annotate_page_rgba(
                &mut rendered.rgba,
                rendered.width,
                rendered.height,
                rendered.page_width,
                rendered.page_height,
                &annotated,
            );
            let annotate_ms = annotate_started.elapsed().as_secs_f64() * 1000.0;

            let encode_png_started = Instant::now();
            let png_bytes = encode_png_rgba(&rendered.rgba, rendered.width, rendered.height)
                .map_err(|error| format!("encode page {} PNG: {error}", rendered.page_number))?;
            let encode_png_ms = encode_png_started.elapsed().as_secs_f64() * 1000.0;

            let image_path = output_path_for_page(output_dir, rendered.page_number, "png");
            let json_path = output_path_for_page(output_dir, rendered.page_number, "json");

            let write_png_started = Instant::now();
            fs::write(&image_path, png_bytes)
                .map_err(|error| format!("write {}: {error}", image_path.display()))?;
            let write_png_ms = write_png_started.elapsed().as_secs_f64() * 1000.0;

            let serialize_json_started = Instant::now();
            let json = NativePageOutput {
                page_number: rendered.page_number,
                width: rendered.width,
                height: rendered.height,
                page_width: rendered.page_width,
                page_height: rendered.page_height,
                dpi,
                detections: &annotated,
            };
            let json = serde_json::to_vec_pretty(&json).map_err(|error| {
                format!("serialize page {} JSON: {error}", rendered.page_number)
            })?;
            let serialize_json_ms = serialize_json_started.elapsed().as_secs_f64() * 1000.0;

            let write_json_started = Instant::now();
            fs::write(&json_path, json)
                .map_err(|error| format!("write {}: {error}", json_path.display()))?;
            let write_json_ms = write_json_started.elapsed().as_secs_f64() * 1000.0;

            let total_page_ms = rendered.page_started.elapsed().as_secs_f64() * 1000.0;

            tracing::event!(
                Level::INFO,
                page_number = rendered.page_number,
                page_count,
                dpi,
                width = rendered.width,
                height = rendered.height,
                page_width = rendered.page_width,
                page_height = rendered.page_height,
                image = %image_path.display(),
                json = %json_path.display(),
                boxes = annotated.len(),
                load_page_ms = rendered.load_page_ms,
                render_ms = rendered.render_ms,
                bitmap_ms = rendered.bitmap_ms,
                detect_ms = batch_detect_ms,
                batch_size = batch_end - batch_start,
                annotate_ms,
                encode_png_ms,
                write_png_ms,
                serialize_json_ms,
                write_json_ms,
                total_page_ms,
                "native page completed"
            );
        }

        batch_start = batch_end;
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

struct RenderedPage {
    page_started: Instant,
    page_number: u32,
    width: u32,
    height: u32,
    page_width: f32,
    page_height: f32,
    rgb: Vec<u8>,
    rgba: Vec<u8>,
    load_page_ms: f64,
    render_ms: f64,
    bitmap_ms: f64,
}

impl RenderedPage {
    /// Borrows this rendered page as model input without copying image buffers.
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

/// Encodes an RGBA page buffer as PNG bytes.
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

/// Builds a stable output file path for one-based page numbers.
fn output_path_for_page(output_dir: &Path, page_number: u32, extension: &str) -> PathBuf {
    output_dir.join(format!("page-{page_number:04}.{extension}"))
}

/// Initializes stderr tracing for native binaries using `RUST_LOG` when present.
fn init_tracing() {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
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
    fn native_cli_detect_uses_default_output_dir_and_dpi() {
        let parsed =
            NativeCli::try_parse_from(["doclayout-detector", "detect", "input.pdf"]).unwrap();

        let NativeCommand::Detect(args) = parsed.command;

        assert_eq!(args.output_dir, Path::new("output"));
        assert_eq!(args.dpi, 96.0);
        assert_eq!(args.batch_size, 1);
    }

    #[test]
    fn native_cli_detect_accepts_explicit_output_dir_and_dpi() {
        let parsed = NativeCli::try_parse_from([
            "doclayout-detector",
            "detect",
            "input.pdf",
            "--output-dir",
            "out",
            "--dpi",
            "150",
            "--batch-size",
            "4",
        ])
        .unwrap();

        let NativeCommand::Detect(args) = parsed.command;

        assert_eq!(args.output_dir, Path::new("out"));
        assert_eq!(args.dpi, 150.0);
        assert_eq!(args.batch_size, 4);
    }

    #[test]
    fn native_cli_detect_rejects_batch_size_outside_supported_range() {
        let too_small = NativeCli::try_parse_from([
            "doclayout-detector",
            "detect",
            "input.pdf",
            "--batch-size",
            "0",
        ]);
        let too_large = NativeCli::try_parse_from([
            "doclayout-detector",
            "detect",
            "input.pdf",
            "--batch-size",
            "5",
        ]);

        assert!(too_small.is_err());
        assert!(too_large.is_err());
    }

    #[test]
    fn output_path_uses_padded_one_based_page_number() {
        assert_eq!(
            output_path_for_page(Path::new("out"), 7, "png"),
            Path::new("out/page-0007.png")
        );
    }
}
