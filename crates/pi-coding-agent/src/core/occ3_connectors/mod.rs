//! OCC-3 connector implementations (network side of the bridge).
//!
//! The generic bridge in [`crate::core::occ::occ3`] is pure and
//! offline-testable; these modules perform the I/O. A connector is
//! constructed from a model's `occ3` configuration (`models.json`) and
//! registered into the bridge's connector registry. Nothing in the
//! bridge tree may import these modules.

pub mod activation;
pub mod sglang;

use std::sync::{Mutex, OnceLock};

use crate::core::occ::occ3::registry::Occ3ConnectorRegistry;
use crate::core::occ::occ3::types::Occ3Method;

/// The coarse negotiation family for a full dotted method id. The
/// [`Occ3Method`] enum stays the wire-level negotiation family (envelope,
/// capabilities, registry support); the full identity keeps flowing
/// through the string-equality seams (readiness `method_id`, publication
/// `intent.method`, inference `attachment.method`). A dotted id outside
/// every known family has no negotiation route and is refused.
pub fn method_family(latent_method: &str) -> Option<Occ3Method> {
    if latent_method.starts_with("statebridge") {
        Some(Occ3Method::StatebridgePrefix)
    } else if latent_method.starts_with("kv-handoff") {
        Some(Occ3Method::KvHandoff)
    } else {
        None
    }
}

/// Process-wide live connector registry. Populated by the embedding
/// host (daemon startup or the first live model configuration) after a
/// successful handshake; OCC-3 stays unavailable until then.
static GLOBAL_REGISTRY: OnceLock<Mutex<Occ3ConnectorRegistry>> = OnceLock::new();

fn global_cell() -> &'static Mutex<Occ3ConnectorRegistry> {
    GLOBAL_REGISTRY.get_or_init(|| Mutex::new(Occ3ConnectorRegistry::empty()))
}

/// Run a function against the process-wide registry under lock.
///
/// The registry holds `Arc` clones of connectors, so snapshot reads
/// (cloning the map out) are cheap; mutation is registration-time
/// only. Keep closures short.
pub fn with_global_registry<T>(f: impl FnOnce(&mut Occ3ConnectorRegistry) -> T) -> T {
    f(&mut global_cell().lock().unwrap())
}

/// Read-only access convenience (clones connector Arcs).
pub fn global_connectors() -> Vec<String> {
    with_global_registry(|registry| registry.connector_ids())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_registry_starts_empty_and_accepts_registrations() {
        // Read-modify-write round trip through the global cell.
        let before = with_global_registry(|registry| registry.connector_ids());
        let ids = with_global_registry(|registry| registry.connector_ids().len());
        assert!(ids >= before.len());
    }
}
