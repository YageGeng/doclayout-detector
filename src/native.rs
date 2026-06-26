use crate::PageImage;
use crate::annotate::{AnnotatedDetection, annotate_page_rgba};
use crate::model::EmbeddedModel;
use crate::pp_doclayout::{PPDocLayoutV3Detector, PPDocLayoutV3Options};
use pdfium::Library;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

pub fn run_cli() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    let Some(input_pdf) = args.get(1) else {
        print_usage(&args[0]);
        return Ok(());
    };
    let Some(output_dir) = args.get(2) else {
        print_usage(&args[0]);
        return Ok(());
    };
    let dpi = match args.get(3) {
        Some(value) => value
            .parse::<f32>()
            .map_err(|error| format!("invalid dpi '{value}': {error}"))?,
        None => 144.0,
    };

    detect_pdf_to_dir(Path::new(input_pdf), Path::new(output_dir), dpi)
}

pub fn detect_pdf_to_dir(input_pdf: &Path, output_dir: &Path, dpi: f32) -> Result<(), String> {
    fs::create_dir_all(output_dir)
        .map_err(|error| format!("create output dir {}: {error}", output_dir.display()))?;

    let model = EmbeddedModel::new().map_err(|error| format!("initialize model: {error}"))?;
    let detector = PPDocLayoutV3Detector::new(model, PPDocLayoutV3Options::default());
    let pdf_bytes = fs::read(input_pdf)
        .map_err(|error| format!("read PDF {}: {error}", input_pdf.display()))?;
    let lib = Library::init();
    let document = lib
        .load_document_from_bytes(&pdf_bytes, None)
        .map_err(|error| format!("load PDF {}: {error}", input_pdf.display()))?;
    let page_count = document.page_count();

    for page_index in 0..page_count {
        let page_number = page_index as u32 + 1;
        let page = document
            .page(page_index)
            .map_err(|error| format!("load page {page_number}: {error}"))?;
        let page_width = page.width();
        let page_height = page.height();
        let bitmap = page
            .render(dpi)
            .map_err(|error| format!("render page {page_number}: {error}"))?;
        let width = bitmap.width() as u32;
        let height = bitmap.height() as u32;
        let rgb = bitmap.to_rgb();
        let mut rgba = bitmap.to_rgba();
        let image = PageImage {
            rgb: &rgb,
            width,
            height,
            page_width,
            page_height,
            dpi,
        };
        let detections = detector
            .detect_page(&image)
            .map_err(|error| format!("detect page {page_number}: {error}"))?;
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
        let png_bytes = encode_png_rgba(&rgba, width, height)
            .map_err(|error| format!("encode page {page_number} PNG: {error}"))?;

        let image_path = output_path_for_page(output_dir, page_number, "png");
        let json_path = output_path_for_page(output_dir, page_number, "json");
        fs::write(&image_path, png_bytes)
            .map_err(|error| format!("write {}: {error}", image_path.display()))?;
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
        fs::write(&json_path, json)
            .map_err(|error| format!("write {}: {error}", json_path.display()))?;

        tracing::info!(
            image = %image_path.display(),
            json = %json_path.display(),
            boxes = annotated.len(),
            "wrote page outputs"
        );
    }

    Ok(())
}

fn print_usage(bin: &str) {
    tracing::info!(
        usage = %format!("{bin} <input.pdf> <output-dir> [dpi]"),
        example = "cargo run --no-default-features --features backend-metal,native-cli -- file.pdf out-metal 144",
        "native CLI usage"
    );
}

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
    fn output_path_uses_padded_one_based_page_number() {
        assert_eq!(
            output_path_for_page(Path::new("out"), 7, "png"),
            Path::new("out/page-0007.png")
        );
    }
}
