use doclayout_detector::pp_doclayout::PPDocLayoutV3Weights;
use std::path::Path;

#[test]
fn pp_doclayout_weights_file_exposes_expected_key_shapes_when_present() {
    let path = Path::new("models/pp_doclayout_v3/model.safetensors");
    if !path.exists() {
        tracing::warn!(
            path = %path.display(),
            "skipping PP-DocLayoutV3 weight shape test; model file is absent"
        );
        return;
    }

    let weights = PPDocLayoutV3Weights::from_file(path).unwrap();

    assert_eq!(
        weights
            .info("model.backbone.model.embedder.stem1.convolution.weight")
            .unwrap()
            .shape,
        vec![32, 3, 3, 3]
    );
    assert_eq!(
        weights.info("model.enc_score_head.weight").unwrap().shape,
        vec![25, 256]
    );
}
