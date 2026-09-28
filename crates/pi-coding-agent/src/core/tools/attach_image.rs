//! Image attachments independent of the Python kernel and execution language.
use super::{path_utils::resolve_read_path, ToolDefinition};
use crate::utils::image_resize::{resize_image, ImageContent, ImageResizeOptions};
use base64::Engine as _;
use pi_agent_core::types::{AgentToolResult, ContentBlock, ToolExecutionMode};
use pi_ai::types::{InputModality, Model};
use serde_json::{json, Value};
use std::{io::Cursor, sync::Arc};
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

const MAX_BYTES: u64 = 20_000_000;
const MAX_PIXELS: u64 = 36_000_000;

pub async fn attach_images(
    cwd: &str,
    paths: &[String],
    signal: CancellationToken,
) -> anyhow::Result<AgentToolResult> {
    anyhow::ensure!(
        !paths.is_empty() && paths.len() <= 8,
        "Provide between 1 and 8 image paths"
    );
    let mut content = Vec::new();
    let mut notes = Vec::new();
    for path in paths {
        anyhow::ensure!(!signal.is_cancelled(), "Image attachment cancelled");
        let path = resolve_read_path(path, cwd);
        anyhow::ensure!(
            tokio::fs::metadata(&path).await?.is_file(),
            "{path} is not a regular file"
        );
        let file = tokio::fs::File::open(&path).await?;
        let metadata = file.metadata().await?;
        anyhow::ensure!(metadata.is_file(), "{path} is not a regular file");
        anyhow::ensure!(
            metadata.len() <= MAX_BYTES,
            "{path} exceeds the 20 MB image limit"
        );
        let mut data = Vec::new();
        file.take(MAX_BYTES + 1).read_to_end(&mut data).await?;
        anyhow::ensure!(
            data.len() as u64 <= MAX_BYTES,
            "{path} exceeds the 20 MB image limit"
        );
        let mime = crate::utils::mime::file_type_from_buffer(&data)
            .ok_or_else(|| anyhow::anyhow!("{path}: expected PNG, JPEG, GIF or WebP"))?;
        let (width, height) = image::ImageReader::new(Cursor::new(&data))
            .with_guessed_format()?
            .into_dimensions()?;
        anyhow::ensure!(
            u64::from(width) * u64::from(height) <= MAX_PIXELS,
            "{path} exceeds the 36 MP image limit"
        );
        let input = ImageContent {
            kind: "image".into(),
            data: base64::engine::general_purpose::STANDARD.encode(data),
            mime_type: mime.into(),
        };
        let image = resize_image(
            &input,
            Some(ImageResizeOptions {
                max_width: Some(1200),
                max_height: Some(1200),
                max_bytes: Some(350_000.0),
                ..Default::default()
            }),
        )
        .await
        .ok_or_else(|| {
            anyhow::anyhow!("{path}: image cannot be decoded or resized for inline display")
        })?;
        notes.push(format!("{path} ({}x{})", image.width, image.height));
        content.push(ContentBlock::Image(pi_ai::types::ImageContent::new(
            image.data,
            image.mime_type,
        )));
    }
    anyhow::ensure!(!signal.is_cancelled(), "Image attachment cancelled");
    content.insert(0, ContentBlock::text(format!("Inline image attachment(s): {}. These images are attached to this tool result and available to the model.", notes.join(", "))));
    Ok(AgentToolResult::new(content, json!({"paths":paths})))
}

pub fn create_attach_image_tool(
    cwd: &str,
    model: Arc<dyn Fn() -> Option<Model> + Send + Sync>,
) -> ToolDefinition<Value> {
    let cwd = cwd.to_owned();
    ToolDefinition {
        name: "attach_image".into(), label: "Image".into(),
        description: "Attach local PNG, JPEG, GIF or WebP files as images to this chat and the model's context. Works without IPython. Use this native tool after generating a chart or screenshot; do not run the Python attach_image CLI through bash or Node. Relative paths use the project directory.".into(),
        parameters: json!({"type":"object","properties":{"paths":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":8}},"required":["paths"]}),
        execution_mode: Some(ToolExecutionMode::Sequential),
        execute: Arc::new(move |_, args, signal, _, _| {
            let cwd = cwd.clone(); let model = model.clone();
            Box::pin(async move {
                let model = model().ok_or_else(|| anyhow::anyhow!("Image attachment session is closed"))?;
                anyhow::ensure!(model.input.contains(&InputModality::Image), "{} does not support images; select a vision-capable model", model.id);
                let paths: Vec<String> = serde_json::from_value(args["paths"].clone())?;
                attach_images(&cwd, &paths, signal.unwrap_or_default()).await
            })
        }),
        ..Default::default()
    }
}
