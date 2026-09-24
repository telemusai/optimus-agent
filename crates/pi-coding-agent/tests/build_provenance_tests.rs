//! Offline build-input tests; no compiler wrapper or external environment pins required.
#[path = "../build_provenance.rs"]
mod build_provenance;
use build_provenance::*;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn put(root: &Path, path: &str, bytes: &[u8]) {
    let path = root.join(path);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}
fn fixture() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    for name in PAYLOAD_FILES { put(tmp.path(), name, name.as_bytes()); }
    for name in ["Cargo.toml", "Cargo.lock", "crates/example/Cargo.toml", "crates/example/build.rs",
                 "crates/example/build_provenance.rs", "crates/example/src/lib.rs", "resources/agent/package.json",
                 "prime-agent-runtime/src/rlm/__init__.py", "prime-agent-runtime/src/rlm/lifecycle.py"] {
        put(tmp.path(), name, name.as_bytes());
    }
    tmp
}

#[test]
fn runtime_hash_uses_sorted_immediate_names_and_raw_bytes_exactly() {
    let tmp = fixture();
    let root = tmp.path();
    put(root, "prime-agent-runtime/src/rlm/__init__.py", b"a\r\n");
    put(root, "prime-agent-runtime/src/rlm/lifecycle.py", b"b\n");
    let mut expected = Sha256::new();
    expected.update(b"__init__.py\0"); expected.update(Sha256::digest(b"a\r\n"));
    expected.update(b"lifecycle.py\0"); expected.update(Sha256::digest(b"b\n"));
    let original = runtime_hash(root).unwrap();
    assert_eq!(original, format!("{:x}", expected.finalize()));
    put(root, "prime-agent-runtime/src/rlm/__init__.py", b"a\n");
    assert_ne!(runtime_hash(root).unwrap(), original, "line endings are raw, never normalized");
    put(root, "prime-agent-runtime/src/rlm/new.py", b"new");
    let added = runtime_hash(root).unwrap();
    fs::remove_file(root.join("prime-agent-runtime/src/rlm/new.py")).unwrap();
    assert_ne!(runtime_hash(root).unwrap(), added);
    fs::remove_file(root.join("prime-agent-runtime/src/rlm/lifecycle.py")).unwrap();
    assert!(runtime_hash(root).is_err());
}

#[test]
fn source_and_payload_hashes_are_relocatable_and_catch_native_helpers_and_new_files() {
    let a = fixture(); let b = fixture();
    let hash = |root: &Path| aggregate(root, &source_files(root).unwrap()).unwrap();
    assert_eq!(hash(a.path()), hash(b.path()));
    let original = hash(a.path());
    put(a.path(), "crates/example/build_provenance.rs", b"changed helper");
    assert_ne!(hash(a.path()), original);
    let before = hash(b.path());
    put(b.path(), "crates/example/src/new.rs", b"new native source");
    assert_ne!(hash(b.path()), before);
    let inputs = source_files(a.path()).unwrap();
    assert!(inputs.iter().all(|name| !Path::new(name).is_absolute() && !name.contains('\\')));
    assert!(inputs.iter().any(|name| name == "Cargo.lock"));
    let watch = watched_paths(a.path());
    assert!(watch.contains(&a.path().join("prime-agent-runtime")), "directory watch notices additions/removals");
    assert!(watch.contains(&a.path().join("crates")));
}

#[test]
fn caches_and_generated_receipts_are_not_build_inputs_but_payload_changes_are() {
    let tmp = fixture(); let root = tmp.path();
    let hash = || aggregate(root, &payload_files(root).unwrap()).unwrap();
    let original = hash();
    for name in ["prime-agent-runtime/.pytest_cache/secret", "prime-agent-runtime/src/rlm/__pycache__/old.pyc",
                 "prime-agent-runtime/.venv/config", "BUILD-PROVENANCE.json", "target/build-receipt.json"] {
        put(root, name, b"ignored");
    }
    assert_eq!(hash(), original);
    put(root, "resources/agent/added.json", b"new payload");
    assert_ne!(hash(), original);
    fs::remove_file(root.join("resources/agent/added.json")).unwrap();
    assert_eq!(hash(), original);
    fs::remove_file(root.join("scripts/optimus-agent")).unwrap();
    assert!(aggregate(root, &payload_files(root).unwrap()).is_err());
}

#[test]
fn fingerprint_binds_inputs_and_rejects_stale_or_empty_environment_pins() {
    assert_eq!(FINGERPRINT_FIELDS.len(), 8);
    let pin = fingerprint(&["source", "runtime", "release"]);
    assert_eq!(pin.len(), 64);
    assert_ne!(pin, fingerprint(&["source", "runtime", "debug"]));
    assert_ne!(pin, fingerprint(&["different", "runtime", "release"]));
    assert!(check_override("OPTIMUS_BUILD_FINGERPRINT", None, &pin).is_ok());
    assert!(check_override("OPTIMUS_BUILD_FINGERPRINT", Some(&pin), &pin).is_ok());
    for stale in ["", "old fingerprint", &"a".repeat(64)] {
        assert!(check_override("OPTIMUS_BUILD_FINGERPRINT", Some(stale), &pin).is_err());
        assert!(check_override("OPTIMUS_RUNTIME_SOURCE_SHA256", Some(stale), &pin).is_err());
    }
}
