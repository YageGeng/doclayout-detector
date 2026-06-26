use doclayout_detector::pp_doclayout::PPDocLayoutV3Label;
use doclayout_detector::{AnnotatedDetection, annotate_page_rgba};

#[test]
fn annotation_maps_pdf_points_to_rendered_pixels() {
    let mut rgba = vec![255; 200 * 100 * 4];
    let detections = [AnnotatedDetection {
        label: PPDocLayoutV3Label::Text,
        confidence: 0.91,
        order: 0,
        x: 10.0,
        y: 20.0,
        width: 30.0,
        height: 10.0,
    }];

    annotate_page_rgba(&mut rgba, 200, 100, 100.0, 50.0, &detections);

    let top_left = ((40 * 200 + 20) * 4) as usize;
    assert_eq!(&rgba[top_left..top_left + 4], &[0x43, 0xA0, 0x47, 255]);

    let bottom_right = ((60 * 200 + 80) * 4) as usize;
    assert_eq!(
        &rgba[bottom_right..bottom_right + 4],
        &[0x43, 0xA0, 0x47, 255]
    );
}

#[test]
fn annotation_ignores_out_of_bounds_edges_without_resizing_buffer() {
    let mut rgba = vec![255; 8 * 8 * 4];
    let original_len = rgba.len();
    let detections = [AnnotatedDetection {
        label: PPDocLayoutV3Label::Table,
        confidence: 0.80,
        order: 0,
        x: -10.0,
        y: -10.0,
        width: 40.0,
        height: 40.0,
    }];

    annotate_page_rgba(&mut rgba, 8, 8, 8.0, 8.0, &detections);

    assert_eq!(rgba.len(), original_len);
    assert_eq!(&rgba[0..4], &[0x00, 0x96, 0x88, 255]);
}

#[test]
fn annotation_uses_pp_doclayout_label_enum_colors() {
    let mut rgba = vec![255; 20 * 20 * 4];
    let detections = [AnnotatedDetection {
        label: PPDocLayoutV3Label::DocTitle,
        confidence: 0.80,
        order: 0,
        x: 2.0,
        y: 2.0,
        width: 8.0,
        height: 8.0,
    }];

    annotate_page_rgba(&mut rgba, 20, 20, 20.0, 20.0, &detections);

    let top_left = ((2 * 20 + 2) * 4) as usize;
    assert_eq!(
        &rgba[top_left..top_left + 4],
        &PPDocLayoutV3Label::DocTitle.debug_color_rgba()
    );
}
