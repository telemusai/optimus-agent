//! Port of packages/ai/src/utils/oauth/oauth-page.ts

// Telemus tesseract geometry, shared with the telemus.ai favicon and embedded
// so local authentication callbacks do not depend on a network asset.
pub const TELEMUS_TESSERACT_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" aria-hidden="true" focusable="false" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round">
  <path d="M16,2.72 L27.4978,9.36 L27.4978,22.64 L16,29.28 L4.5022,22.64 L4.5022,9.36 Z"/>
  <path d="M16,9.36 L21.7489,12.68 L21.7489,19.32 L16,22.64 L10.2511,19.32 L10.2511,12.68 Z"/>
  <path d="M16,2.72 L16,9.36 M27.4978,9.36 L21.7489,12.68 M27.4978,22.64 L21.7489,19.32 M16,29.28 L16,22.64 M4.5022,22.64 L10.2511,19.32 M4.5022,9.36 L10.2511,12.68"/>
  <path d="M10.2511,12.68 L16,16 L21.7489,12.68 M10.2511,19.32 L16,16 L21.7489,19.32 M16,16 L16,22.64"/>
</svg>"#;

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RenderPageOptions {
    pub title: String,
    pub heading: String,
    pub message: String,
    pub details: Option<String>,
}

fn render_page(options: &RenderPageOptions) -> String {
    let title = escape_html(&options.title);
    let heading = escape_html(&options.heading);
    let message = escape_html(&options.message);
    let details = options.details.as_ref().map(|details| escape_html(details));

    let details_block = match details {
        Some(details) => format!("<div class=\"details\">{}</div>", details),
        None => String::new(),
    };

    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --text: #ffffff;
      --brand: #94cf14;
      --text-dim: #a1a1aa;
      --page-bg: #000000;
      --font-sans: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }}
    * {{ box-sizing: border-box; }}
    html {{ color-scheme: dark; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: var(--page-bg);
      color: var(--text);
      font-family: var(--font-sans);
      text-align: center;
    }}
    main {{
      width: 100%;
      max-width: 560px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }}
    .logo {{
      width: 72px;
      height: 72px;
      display: block;
      margin-bottom: 16px;
      color: var(--brand);
    }}
    .logo svg {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 28px;
      line-height: 1.15;
      font-weight: 650;
      color: var(--text);
    }}
    p {{
      margin: 0;
      line-height: 1.7;
      color: var(--text-dim);
      font-size: 15px;
    }}
    .brand {{
      margin-bottom: 28px;
    }}
    .brand strong {{
      color: var(--text);
      font-weight: 650;
    }}
    .details {{
      margin-top: 16px;
      font-family: var(--font-mono);
      font-size: 13px;
      color: var(--text-dim);
      white-space: pre-wrap;
      word-break: break-word;
    }}
  </style>
</head>
<body>
  <main>
    <div class="logo" aria-hidden="true">{svg}</div>
    <p class="brand"><strong>Optimus</strong> by Telemus AI</p>
    <h1>{heading}</h1>
    <p>{message}</p>
    {details_block}
  </main>
</body>
</html>"#,
        title = title,
        svg = TELEMUS_TESSERACT_SVG,
        heading = heading,
        message = message,
        details_block = details_block,
    )
}

pub fn oauth_success_html(message: &str) -> String {
    render_page(&RenderPageOptions {
        title: "Optimus authentication successful".to_string(),
        heading: "Authentication successful".to_string(),
        message: message.to_string(),
        details: None,
    })
}

pub fn oauth_error_html(message: &str, details: Option<&str>) -> String {
    render_page(&RenderPageOptions {
        title: "Optimus authentication failed".to_string(),
        heading: "Authentication failed".to_string(),
        message: message.to_string(),
        details: details.map(str::to_string),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_page_contains_heading_and_message() {
        let html = oauth_success_html("Anthropic authentication completed.");
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("<title>Optimus authentication successful</title>"));
        assert!(html.contains("<h1>Authentication successful</h1>"));
        assert!(html.contains("<p>Anthropic authentication completed.</p>"));
        assert!(html.contains(TELEMUS_TESSERACT_SVG));
        assert!(html.contains("<strong>Optimus</strong> by Telemus AI"));
        assert!(!html.contains("Prime Intellect"));
        assert!(!html.contains("<div class=\"details\">"));
    }

    #[test]
    fn error_page_renders_escaped_details() {
        let html = oauth_error_html("Failed <badly>", Some("Error: access_denied & 'quoted'"));
        assert!(html.contains("<title>Optimus authentication failed</title>"));
        assert!(html.contains("<strong>Optimus</strong> by Telemus AI"));
        assert!(html.contains("<h1>Authentication failed</h1>"));
        assert!(html.contains("<p>Failed &lt;badly&gt;</p>"));
        assert!(html.contains("<div class=\"details\">Error: access_denied &amp; &#39;quoted&#39;</div>"));
    }
}
