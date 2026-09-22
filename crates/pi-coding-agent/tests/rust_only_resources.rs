//! Native assets must resolve in a source checkout and in a relocated bundle.
use pi_coding_agent::config;
use std::fs;
use std::path::Path;
use std::process::Command;

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let target = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn source_checkout_resolves_resources_without_typescript() {
    let package = config::get_package_dir();
    assert!(
        Path::new(&package).ends_with("resources/agent"),
        "{package}"
    );
    for theme in ["prime.json", "dark.json", "light.json"] {
        assert!(Path::new(&config::get_themes_dir()).join(theme).is_file());
    }
    assert!(Path::new(&config::get_bundled_skills_dir())
        .join("edit/pyproject.toml")
        .is_file());
    assert!(Path::new(&config::get_export_template_dir())
        .join("template.html")
        .is_file());
    assert!(Path::new(&config::get_docs_path())
        .join("providers.md")
        .is_file());
}

#[test]
fn relocated_binary_uses_bundle_assets_and_exports_html() {
    let scratch = tempfile::tempdir().unwrap();
    let bundle = scratch.path().join("bundle with spaces");
    let assets = bundle.join("resources/agent");
    copy_tree(Path::new(&config::get_package_dir()), &assets);
    // A distinct template proves the executable prefers its bundle to the checkout.
    let template = assets.join("src/core/export-html/template.html");
    let html = fs::read_to_string(&template).unwrap() + "\n<!-- relocated-resource-probe -->\n";
    fs::write(template, html).unwrap();
    let bin = bundle.join("bin");
    fs::create_dir(&bin).unwrap();
    let executable = bin.join(if cfg!(windows) {
        "optimus-rust.exe"
    } else {
        "optimus-rust"
    });
    fs::copy(env!("CARGO_BIN_EXE_optimus-rust"), &executable).unwrap();
    let command = || {
        let mut cmd = Command::new(&executable);
        cmd.current_dir(scratch.path())
            .env_remove("PI_PACKAGE_DIR")
            .env_remove("PI_CODING_AGENT_MODULE_DIR")
            .env(
                "PRIME_AGENT_CODING_AGENT_DIR",
                scratch.path().join("profile"),
            )
            .env("PI_OFFLINE", "1");
        cmd
    };
    let version = command().arg("--version").output().unwrap();
    assert!(
        version.status.success(),
        "{}",
        String::from_utf8_lossy(&version.stderr)
    );
    let version_text = format!(
        "{}{}",
        String::from_utf8_lossy(&version.stdout),
        String::from_utf8_lossy(&version.stderr)
    );
    assert!(
        version_text.contains(env!("CARGO_PKG_VERSION")),
        "{version_text}"
    );
    let session = scratch.path().join("sample.jsonl");
    fs::write(&session, concat!(
        "{\"type\":\"session\",\"version\":3,\"id\":\"probe\",\"timestamp\":\"2026-09-23T00:00:00Z\",\"cwd\":\".\"}\n",
        "{\"type\":\"message\",\"id\":\"m1\",\"parentId\":null,\"timestamp\":\"2026-09-23T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"Native bundle export\",\"timestamp\":1}}\n"
    )).unwrap();
    let export = command()
        .args(["session", "export"])
        .arg(&session)
        .output()
        .unwrap();
    assert!(
        export.status.success(),
        "{}",
        String::from_utf8_lossy(&export.stderr)
    );
    let html = fs::read_to_string(scratch.path().join("prime-agent-session-sample.html")).unwrap();
    assert!(html.to_lowercase().contains("<!doctype html>"));
    assert!(html.contains("<!-- relocated-resource-probe -->"));
    assert!(!html.contains("{{TEMPLATE_JS}}"));
}
