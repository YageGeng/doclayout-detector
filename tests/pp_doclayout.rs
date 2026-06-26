use doclayout_detector::AnnotatedDetection;
use doclayout_detector::LayoutError;
use doclayout_detector::PageImage;
use doclayout_detector::pp_doclayout::{
    PP_DOCLAYOUT_V3_IMAGE_SIZE, PPDocLayoutV3Config, PPDocLayoutV3Detection, PPDocLayoutV3Detector,
    PPDocLayoutV3Inference, PPDocLayoutV3Label, PPDocLayoutV3Options, PPDocLayoutV3OwnedOutputs,
    PPDocLayoutV3RawOutputs, decode_box_detections, resize_rgb_to_chw_f32,
};

fn page_image<'a>(rgb: &'a [u8], width: u32, height: u32) -> PageImage<'a> {
    PageImage {
        rgb,
        width,
        height,
        page_width: width as f32 / 2.0,
        page_height: height as f32 / 2.0,
        dpi: 144.0,
    }
}

#[test]
fn pp_doclayout_label_converts_from_class_id() {
    assert_eq!(
        PPDocLayoutV3Label::try_from(0),
        Ok(PPDocLayoutV3Label::Abstract)
    );
    assert_eq!(
        PPDocLayoutV3Label::try_from(21),
        Ok(PPDocLayoutV3Label::Table)
    );
    assert_eq!(
        PPDocLayoutV3Label::try_from(24),
        Ok(PPDocLayoutV3Label::VisionFootnote)
    );
    assert!(PPDocLayoutV3Label::try_from(25).is_err());
}

#[test]
fn pp_doclayout_label_serializes_to_official_snake_case() {
    assert_eq!(
        serde_json::to_string(&PPDocLayoutV3Label::DocTitle).unwrap(),
        "\"doc_title\""
    );
    assert_eq!(
        serde_json::to_string(&PPDocLayoutV3Label::AsideText).unwrap(),
        "\"aside_text\""
    );
    assert_eq!(
        serde_json::to_string(&PPDocLayoutV3Label::VisionFootnote).unwrap(),
        "\"vision_footnote\""
    );
}

#[test]
fn pp_doclayout_preprocess_resizes_without_letterbox() {
    let rgb = [
        255, 0, 0, //
        0, 255, 0, //
        0, 0, 255, //
        255, 255, 255,
    ];
    let image = page_image(&rgb, 2, 2);

    let input = resize_rgb_to_chw_f32(&image, 4).unwrap();

    assert_eq!(input.len(), 3 * 4 * 4);
    assert_eq!(input[0], 1.0);
    assert_eq!(input[16], 0.0);
    assert_eq!(input[32], 0.0);
    assert_eq!(input[3], 0.0);
    assert_eq!(input[19], 1.0);
    assert_eq!(input[35], 0.0);
    assert_eq!(input[12], 0.0);
    assert_eq!(input[28], 0.0);
    assert_eq!(input[44], 1.0);
    assert_eq!(input[15], 1.0);
    assert_eq!(input[31], 1.0);
    assert_eq!(input[47], 1.0);
}

#[test]
fn pp_doclayout_box_postprocess_applies_sigmoid_topk_scale_and_order() {
    let rgb = vec![255; 1600 * 1000 * 3];
    let image = page_image(&rgb, 1600, 1000);
    let mut logits = vec![-20.0; 300 * 25];
    let mut pred_boxes = vec![0.0; 300 * 4];
    let mut order_logits = vec![0.0; 300 * 300];

    logits[7 * 25 + 21] = 5.0;
    pred_boxes[7 * 4] = 0.50;
    pred_boxes[7 * 4 + 1] = 0.50;
    pred_boxes[7 * 4 + 2] = 0.25;
    pred_boxes[7 * 4 + 3] = 0.20;

    logits[11 * 25 + 22] = 4.0;
    pred_boxes[11 * 4] = 0.25;
    pred_boxes[11 * 4 + 1] = 0.20;
    pred_boxes[11 * 4 + 2] = 0.10;
    pred_boxes[11 * 4 + 3] = 0.10;

    order_logits[7 * 300 + 11] = -8.0;

    let detections = decode_box_detections(
        &PPDocLayoutV3RawOutputs {
            logits_shape: [1, 300, 25],
            logits: &logits,
            pred_boxes_shape: [1, 300, 4],
            pred_boxes: &pred_boxes,
            order_logits_shape: Some([1, 300, 300]),
            order_logits: Some(&order_logits),
        },
        &image,
        0.5,
    )
    .unwrap();

    assert_eq!(detections.len(), 2);
    assert_eq!(detections[0].label, PPDocLayoutV3Label::Text);
    assert_eq!(detections[1].label, PPDocLayoutV3Label::Table);
    assert!((detections[1].x - 300.0).abs() < 0.001);
    assert!((detections[1].y - 200.0).abs() < 0.001);
    assert!((detections[1].width - 200.0).abs() < 0.001);
    assert!((detections[1].height - 100.0).abs() < 0.001);
}

#[test]
fn pp_doclayout_uses_official_input_size() {
    assert_eq!(PP_DOCLAYOUT_V3_IMAGE_SIZE, 800);
}

#[test]
fn pp_doclayout_config_matches_huggingface_model_card_contract() {
    let config = PPDocLayoutV3Config::default();

    assert_eq!(config.image_size, 800);
    assert_eq!(config.num_queries, 300);
    assert_eq!(config.num_classes, 25);
    assert_eq!(config.d_model, 256);
    assert_eq!(config.decoder_layers, 6);
    assert_eq!(config.decoder_attention_heads, 8);
    assert_eq!(config.encoder_in_channels, [512, 1024, 2048]);
    assert_eq!(config.feature_strides, [8, 16, 32]);
}

#[test]
fn pp_doclayout_detection_converts_to_annotated_detection() {
    let detection = PPDocLayoutV3Detection {
        label: PPDocLayoutV3Label::DocTitle,
        confidence: 0.75,
        order: 3,
        x: 10.0,
        y: 20.0,
        width: 30.0,
        height: 40.0,
    };

    let annotated = AnnotatedDetection::from(&detection);

    assert_eq!(annotated.label, PPDocLayoutV3Label::DocTitle);
    assert_eq!(annotated.confidence, 0.75);
    assert_eq!(annotated.order, 3);
    assert_eq!(annotated.x, 10.0);
}

#[test]
fn pp_doclayout_detection_converts_to_layout_detection_with_order() {
    let detection = PPDocLayoutV3Detection {
        label: PPDocLayoutV3Label::Text,
        confidence: 0.88,
        order: 9,
        x: 11.0,
        y: 12.0,
        width: 13.0,
        height: 14.0,
    };

    let layout = doclayout_detector::LayoutDetection::from(detection);

    assert_eq!(layout.order, 9);
}

#[test]
fn pp_doclayout_detector_runs_preprocess_inference_and_postprocess() {
    struct FakeInference;

    impl PPDocLayoutV3Inference for FakeInference {
        fn infer(&self, input: &[f32]) -> Result<PPDocLayoutV3OwnedOutputs, LayoutError> {
            assert_eq!(
                input.len(),
                3 * PP_DOCLAYOUT_V3_IMAGE_SIZE as usize * PP_DOCLAYOUT_V3_IMAGE_SIZE as usize
            );
            let mut logits = vec![-20.0; 300 * 25];
            let mut pred_boxes = vec![0.0; 300 * 4];
            logits[5 * 25 + 22] = 6.0;
            pred_boxes[5 * 4] = 0.50;
            pred_boxes[5 * 4 + 1] = 0.50;
            pred_boxes[5 * 4 + 2] = 0.50;
            pred_boxes[5 * 4 + 3] = 0.50;

            Ok(PPDocLayoutV3OwnedOutputs {
                logits_shape: [1, 300, 25],
                logits,
                pred_boxes_shape: [1, 300, 4],
                pred_boxes,
                order_logits_shape: None,
                order_logits: None,
            })
        }
    }

    let rgb = vec![255; 20 * 10 * 3];
    let image = page_image(&rgb, 20, 10);
    let detector = PPDocLayoutV3Detector::new(FakeInference, PPDocLayoutV3Options::default());

    let detections = detector.detect_page(&image).unwrap();

    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].label, PPDocLayoutV3Label::Text);
    assert!((detections[0].x - 2.5).abs() < 0.001);
    assert!((detections[0].height - 2.5).abs() < 0.001);
}
