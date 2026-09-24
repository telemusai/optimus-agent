//! Deterministic raw-byte build inputs. Shared by build.rs and offline tests.
use sha2::{Digest, Sha256};
use std::{fs, io, path::{Path, PathBuf}};

pub const SCHEMA: &str = "optimus.build-provenance.v1";
pub const PAYLOAD_FILES: &[&str] = &[
    "scripts/optimus-agent", "scripts/launch-with-jev-env.py", "scripts/rust_release.py",
    "install.sh", "LICENSE", "README.md",
];
pub const FINGERPRINT_FIELDS: &[&str] = &[
    "sourceTreeSha256", "payloadSourceSha256", "runtimeSourceSha256", "version",
    "target", "profile", "rustc", "buildOptionsSha256",
];

fn ignored(name: &str) -> bool {
    matches!(name, "__pycache__" | ".venv" | ".pytest_cache" | ".mypy_cache" | ".ruff_cache") || name.ends_with(".pyc") || name.ends_with(".egg-info")
}

fn walk(root: &Path, directory: &Path, files: &mut Vec<String>) -> io::Result<()> {
    if !fs::symlink_metadata(directory)?.is_dir() { return Err(io::Error::other("build input directory must not be a symlink")); }
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name().into_string().map_err(|_| io::Error::other("non-UTF-8 build input"))?;
        if ignored(&name) { continue; }
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_symlink() { return Err(io::Error::other(format!("symlink build input: {}", path.display()))); }
        if kind.is_dir() { walk(root, &path, files)?; }
        else if kind.is_file() {
            files.push(path.strip_prefix(root).unwrap().to_str().ok_or_else(|| io::Error::other("non-UTF-8 build input"))?.replace('\\', "/"));
        }
    }
    Ok(())
}

pub fn payload_files(root: &Path) -> io::Result<Vec<String>> {
    let mut files: Vec<String> = PAYLOAD_FILES.iter().map(|name| (*name).into()).collect();
    for directory in ["resources", "prime-agent-runtime"] { walk(root, &root.join(directory), &mut files)?; }
    files.sort();
    Ok(files)
}

pub fn source_files(root: &Path) -> io::Result<Vec<String>> {
    let mut files = payload_files(root)?;
    files.extend(["Cargo.toml".into(), "Cargo.lock".into()]);
    for entry in fs::read_dir(root.join("crates"))? {
        let entry = entry?;
        if entry.file_type()?.is_symlink() { return Err(io::Error::other("symlink crate input")); }
        let path = entry.path();
        if !path.is_dir() || !path.join("Cargo.toml").is_file() { continue; }
        for name in ["Cargo.toml", "build.rs", "build_provenance.rs"] {
            let file = path.join(name);
            if file.is_file() { files.push(file.strip_prefix(root).unwrap().to_str().unwrap().replace('\\', "/")); }
        }
        walk(root, &path.join("src"), &mut files)?;
    }
    files.sort();
    files.dedup();
    Ok(files)
}

pub fn aggregate(root: &Path, files: &[String]) -> io::Result<String> {
    let mut digest = Sha256::new();
    for name in files {
        digest.update(name.as_bytes());
        digest.update([0]);
        let path = root.join(name);
        if !fs::symlink_metadata(&path)?.is_file() { return Err(io::Error::other("build input must be a regular file")); }
        digest.update(Sha256::digest(fs::read(path)?));
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn runtime_hash(root: &Path) -> io::Result<String> {
    let directory = root.join("prime-agent-runtime/src/rlm");
    let mut names = Vec::new();
    for entry in fs::read_dir(&directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext|
            ext == "py" || (cfg!(windows) && ext.eq_ignore_ascii_case("py"))) {
            if !entry.file_type()?.is_file() { return Err(io::Error::other("runtime source must be a regular file")); }
            names.push(entry.file_name().into_string().map_err(|_| io::Error::other("non-UTF-8 runtime file"))?);
        }
    }
    names.sort();
    if !names.iter().any(|name| name == "lifecycle.py") || !names.iter().any(|name| name == "__init__.py") {
        return Err(io::Error::other("runtime lifecycle sources missing"));
    }
    aggregate(&directory, &names)
}

pub fn fingerprint(values: &[&str]) -> String {
    let mut digest = Sha256::new();
    digest.update(SCHEMA.as_bytes());
    digest.update([0]);
    for value in values { digest.update(value.as_bytes()); digest.update([0]); }
    format!("{:x}", digest.finalize())
}

pub fn check_override(name: &str, supplied: Option<&str>, actual: &str) -> io::Result<()> {
    if supplied.is_some_and(|value| value != actual) {
        return Err(io::Error::other(format!("{name} does not match current build inputs; remove stale provenance override")));
    }
    Ok(())
}

pub fn watched_paths(root: &Path) -> Vec<PathBuf> {
    ["Cargo.toml", "Cargo.lock", "crates", "resources", "prime-agent-runtime"]
        .iter().chain(PAYLOAD_FILES.iter()).map(|name| root.join(name)).collect()
}
