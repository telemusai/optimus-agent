//! Native attachment validation without a kernel or provider.
use pi_coding_agent::core::tools::attach_image::attach_images;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn image_batches_reject_invalid_files_empty_batches_and_cancellation() {
    let root = tempfile::tempdir().unwrap();
    let cwd = root.path().to_str().unwrap();
    image::RgbaImage::from_pixel(8, 8, image::Rgba([255, 0, 0, 255]))
        .save(root.path().join("red.png"))
        .unwrap();
    std::fs::write(root.path().join("broken.png"), b"\x89PNG\r\n\x1a\ninvalid").unwrap();
    assert!(attach_images(cwd, &[], CancellationToken::new())
        .await
        .is_err());
    assert!(attach_images(
        cwd,
        &["red.png".into(), "missing.png".into()],
        CancellationToken::new()
    )
    .await
    .is_err());
    assert!(
        attach_images(cwd, &["broken.png".into()], CancellationToken::new())
            .await
            .is_err()
    );
    assert!(attach_images(cwd, &[".".into()], CancellationToken::new())
        .await
        .is_err());
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert!(attach_images(cwd, &["red.png".into()], cancelled)
        .await
        .unwrap_err()
        .to_string()
        .contains("cancelled"));
    let huge = std::fs::File::create(root.path().join("huge.png")).unwrap();
    huge.set_len(20_000_001).unwrap();
    assert!(
        attach_images(cwd, &["huge.png".into()], CancellationToken::new())
            .await
            .unwrap_err()
            .to_string()
            .contains("20 MB")
    );
}

#[tokio::test]
async fn image_attachment_preserves_small_png_pixels_and_bounds_large_images() {
    use base64::Engine as _;
    let root = tempfile::tempdir().unwrap();
    let cwd = root.path().to_str().unwrap();
    for (name, size) in [("small.png", 8), ("large.png", 1400)] {
        image::RgbaImage::from_pixel(size, size, image::Rgba([12, 240, 130, 255]))
            .save(root.path().join(name))
            .unwrap();
        let result = attach_images(cwd, &[name.into()], CancellationToken::new())
            .await
            .unwrap();
        let value = serde_json::to_value(result).unwrap();
        let image = value["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["type"] == "image")
            .unwrap();
        let data = image["data"].as_str().unwrap();
        assert!(data.len() <= 350_000);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert!(decoded.width() <= 1200 && decoded.height() <= 1200);
        if name == "small.png" {
            assert_eq!(decoded.to_rgba8().get_pixel(0, 0).0, [12, 240, 130, 255]);
        }
    }
}

#[tokio::test]
#[ignore = "requires OPTIMUS_CLANG_REPL pointing to LLVM clang-repl"]
async fn clang_timeout_and_cancellation_reset_only_the_cpp_workspace() {
    use pi_coding_agent::core::tools::clang::ClangRuntime;
    let root = tempfile::tempdir().unwrap();
    let cwd = root.path().to_str().unwrap();
    let runtime = std::sync::Arc::new(ClangRuntime::default());
    runtime
        .execute(cwd, "int prior = 5;", 10.0, None, None)
        .await
        .unwrap();
    let error = runtime
        .execute(
            cwd,
            "auto hang = [] { while (true) {} return 0; }();",
            0.2,
            None,
            None,
        )
        .await
        .unwrap_err();
    assert!(
        error.contains("timed out") && error.contains("reset"),
        "{error}"
    );
    let result = runtime
        .execute(cwd, "int fresh = prior;", 10.0, None, None)
        .await
        .unwrap();
    assert_eq!(
        result.is_error,
        Some(true),
        "timed-out workspace must not retain prior globals"
    );
    let signal = CancellationToken::new();
    let work = runtime.execute(
        cwd,
        "auto hang2 = [] { while (true) {} return 0; }();",
        10.0,
        Some(signal.clone()),
        None,
    );
    let stop = async {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        signal.cancel();
    };
    let (result, ()) = tokio::join!(work, stop);
    assert!(result.unwrap_err().contains("cancelled"));
    let result = runtime
        .execute(
            cwd,
            "auto final = [] { std::printf(\"CPP_RECOVERED\\n\"); return 0; }();",
            10.0,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(result.is_error, Some(false));
    assert!(serde_json::to_string(&result)
        .unwrap()
        .contains("CPP_RECOVERED"));
    runtime.dispose().await;
}
