"""Deployment-side capability advertisement: the registry must be advertised.

The capabilities route is the deployment-facing surface; its "methods" key
must list the method ids the installed registry advertises (build
capability), in every startup state, while "supported" stays false: the
server never self-certifies live proof.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest


@pytest.fixture(autouse=True)
def machine():
    from occ3_sglang import startup_state

    startup_state.reset()
    yield startup_state
    startup_state.reset()


def test_capabilities_advertise_the_registered_methods(stub_http_server, stub_tm):
    from occ3_sglang import native_profile
    from occ3_sglang import startup_state

    stub_http_server._global_state = NS(tokenizer_manager=stub_tm)
    payload = startup_state.capabilities_payload()
    assert payload["methods"] == list(native_profile.advertised_methods())
    assert native_profile.CALIBRATED_METHOD in payload["methods"]
    # Advertisement is capability, never live proof.
    assert payload["supported"] is False


def test_capabilities_advertise_methods_in_every_state(stub_http_server, stub_tm):
    from occ3_sglang import native_profile
    from occ3_sglang import startup_state

    stub_http_server._global_state = NS(tokenizer_manager=stub_tm)
    startup_state.record_declined("source_digest_mismatch", stage="install",
                                  attempt="0123abcd", rollback_completed=True)
    payload = startup_state.capabilities_payload()
    assert payload["methods"] == list(native_profile.advertised_methods())
    assert payload["supported"] is False
    assert payload["reason"] == "native_install_declined"
