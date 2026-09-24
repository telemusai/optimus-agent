mod build_provenance;
use build_provenance::*;
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf, process::Command};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("../..").canonicalize()?;
    for path in watched_paths(&root) { println!("cargo:rerun-if-changed={}", path.display()); }
    for name in ["OPTIMUS_BUILD_FINGERPRINT", "OPTIMUS_RUNTIME_SOURCE_SHA256", "CARGO_ENCODED_RUSTFLAGS", "PI_BUILD_ID"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let output = Command::new(env::var("RUSTC")?).arg("--version").output()?;
    if !output.status.success() { return Err("cannot inspect Rust compiler version".into()); }
    let rustc = String::from_utf8(output.stdout)?.trim().to_owned();
    let mut options: Vec<(String, String)> = env::vars().filter(|(key, _)|
        key.starts_with("CARGO_FEATURE_") || matches!(key.as_str(), "OPT_LEVEL" | "DEBUG" | "CARGO_ENCODED_RUSTFLAGS" | "PI_BUILD_ID")).collect();
    options.sort();
    let build_options = format!("{:x}", Sha256::digest(serde_json::to_vec(&options)?));
    let source = aggregate(&root, &source_files(&root)?)?;
    let payload = aggregate(&root, &payload_files(&root)?)?;
    let runtime = runtime_hash(&root)?;
    let version = env::var("CARGO_PKG_VERSION")?;
    let target = env::var("TARGET")?;
    let profile = env::var("PROFILE")?;
    let values = [&source[..], &payload, &runtime, &version, &target, &profile, &rustc, &build_options];
    let pin = fingerprint(&values);
    check_override("OPTIMUS_BUILD_FINGERPRINT", env::var("OPTIMUS_BUILD_FINGERPRINT").ok().as_deref(), &pin)?;
    check_override("OPTIMUS_RUNTIME_SOURCE_SHA256", env::var("OPTIMUS_RUNTIME_SOURCE_SHA256").ok().as_deref(), &runtime)?;
    let mut receipt = serde_json::Map::new();
    receipt.insert("schema".into(), SCHEMA.into());
    receipt.insert("buildFingerprint".into(), pin.clone().into());
    for (field, value) in FINGERPRINT_FIELDS.iter().zip(values) { receipt.insert((*field).into(), value.into()); }
    let bytes = serde_json::to_vec_pretty(&receipt)?;
    let destination = PathBuf::from(env::var("OUT_DIR")?).join("build-provenance.json");
    if fs::read(&destination).ok().as_deref() != Some(bytes.as_slice()) { fs::write(destination, bytes)?; }
    println!("cargo:rustc-env=OPTIMUS_BUILD_FINGERPRINT={pin}");
    println!("cargo:rustc-env=OPTIMUS_RUNTIME_SOURCE_SHA256={runtime}");
    Ok(())
}
