"""Capture reservation: a session row is never retracted.

These tests run on the scheduler model of test_capture_hold and have the same
limits: this is not a SGLang, GPU, NCCL or TP8 runtime proof. The session is
a stand-in. The tests show that a session row is left alone. They do not show
that the pinned session code could resume a retracted row.
"""
from types import SimpleNamespace as NS

import pytest

from occ3_sglang import latent_types, latent_worker as w
from test_capture_hold import (  # noqa: F401  (sources and world are fixtures)
    KNOBS, Rank, Sender, cap, generate, hold_gathers, once, plain, reference, retractions, sources,
    ticks, world)

SESSION = "session-1"


@pytest.fixture
def action(monkeypatch):
    """Verdict of the real resolution policy for the given in-flight rows."""
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(w, "time", NS(monotonic=lambda: 100.0))
    return lambda preempt, rows: w._capture_hold_action(
        NS(waiting_queue=[]), NS(deadline=101.0),
        latent_types.CaptureHoldSettings(10.0, 2.0, preempt), rows)


@pytest.mark.parametrize("rows, expected", [
    ([plain(session=object())], "decline"),
    ([plain(), plain(session=object())], "decline"),
    ([plain(session=object()), plain()], "decline"),
    ([plain(session=())], "decline"),
    ([plain(session=None)], "retract"),
    ([plain(), plain()], "retract"),
], ids=["session", "plain_then_session", "session_then_plain", "falsy_session_object",
        "session_none", "no_session_field"])
def test_retract_policy_declines_when_a_row_has_a_session(action, rows, expected):
    """One session row declines for every row. Only None is no session."""
    assert action("retract", rows) == expected
    assert action("off", rows) == "decline"


class Session:
    """Stand-in for the pinned legacy Session. create_req is the one call that
    the exact session route of handle_generate_request makes. Like the pinned
    method, it binds the new row to its session."""
    streaming = close_on_finish = False

    def __init__(self, rank):
        self.rank = rank

    def create_req(self, recv_req, tokenizer, vocab_size, **kw):
        row = self.rank._row(recv_req.rid, recv_req.input_text, recv_req.input_ids,
                             recv_req.sampling_params)
        row.session = self
        return row


@pytest.fixture
def session_world(world, monkeypatch):
    """World whose ranks send a request with session_params through the exact
    session route of handle_generate_request."""
    recv = Rank.recv

    def recv_with_session(self, request):
        received = recv(self, request)
        received.session_params = getattr(request, "session_params", None)
        return received

    monkeypatch.setattr(Rank, "recv", recv_with_session)

    def make(**kw):
        sim = world(**kw)
        sim.each(lambda rank: setattr(rank, "session_controller", {SESSION: Session(rank)}))
        return sim

    return make


def in_session(rid, new_tokens):
    request = generate(rid, new_tokens)
    request.session_params = NS(id=SESSION)
    return request


@pytest.mark.parametrize("others", [(), ("plain",)], ids=["session_only", "with_a_plain_row"])
def test_retract_mode_never_retracts_while_a_session_row_decodes(session_world, monkeypatch, others):
    """Opt-in retract, four ranks. A session request decodes past the drain.
    No row is retracted, also not a plain row that runs with it. The capture
    gets the busy refusal at the drain bound, exactly as with preempt off."""
    sent, drain, long = ticks(1.0), ticks(2.0), ticks(6.0)
    views = {}
    for preempt in ("retract", "off"):
        monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", preempt)
        sim = session_world(ranks=4)
        sim.run(8.0, once(0, in_session("session", long), *(generate(rid, long) for rid in others)),
                once(1.0, cap(4)), once(1.25, generate("later", 4)))
        for rank in sim.ranks:
            assert rank.row("session").session is rank.session_controller[SESSION]
            assert all(rank.row(rid).session is None for rid in (*others, "later"))
        assert sim.signals == [(sent + drain, "occ3_capture_hold_1")]
        assert hold_gathers(sim) == [("park", "rid", True), ("resolve", "rid", "decline")]
        assert sim.refusals("rid") == ["occ3_busy_batch"] and sim.first_run("rid") is None
        assert retractions(sim) == {} and sim.record().state == "reserved"
        assert sim.first_run("later") == sent + drain  # new prefills are free at the decline
        for rid, count in (("session", long), ("later", 4), *((rid, long) for rid in others)):
            assert sim.stream(rid) == reference(rid, count) and sim.done(rid)
        sim.assert_clean()
        views[preempt] = [rank.view() for rank in sim.ranks]
    assert views["retract"] == views["off"]


def test_the_same_request_without_a_session_is_retracted(session_world, monkeypatch):
    """Control for the test above: the session is the only difference."""
    monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", "retract")
    sim = session_world(ranks=4)
    sim.run(8.0, once(0, generate("session", ticks(6.0))), once(1.0, cap(4)),
            once(1.25, generate("later", 4)))
    assert hold_gathers(sim) == [("park", "rid", True), ("resolve", "rid", "retract")]
    assert retractions(sim) == {"session": 1} and sim.refusals("rid") == []
    assert sim.first_run("rid") == ticks(1.0) + ticks(2.0)
    for rid, count in (("session", ticks(6.0)), ("later", 4), ("rid", 4)):
        assert sim.stream(rid) == reference(rid, count) and sim.done(rid)
    sim.assert_clean()


def test_a_capture_declined_for_a_session_row_is_admitted_on_resend(session_world, monkeypatch):
    """The bounded path still serves the capture under retract: the frontend
    sends the same request again and it is admitted in the first iteration
    after the session row ends. Nothing is retracted."""
    monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", "retract")
    sim = session_world()
    sent, decode = ticks(1.0), ticks(6.5)
    sender = Sender(cap(8), at=1.0)
    sim.run(10.0, once(0, in_session("session", sent + decode + 1)), sender)
    assert len(sender.sent_at) == 3 and sim.refusals("rid") == ["occ3_busy_batch"] * 2
    assert hold_gathers(sim) == [("park", "rid", True), ("resolve", "rid", "decline"),
                                 ("park", "rid", True), ("resolve", "rid", "decline"),
                                 ("park", "rid", True), ("resolve", "rid", "admit")]
    assert sim.first_run("rid") == sim.last_run("session") + 1 and retractions(sim) == {}
    assert sim.stream("session") == reference("session", sent + decode + 1) and sim.done("session")
    assert sim.stream("rid") == reference("rid", 8) and sim.done("rid")
    sim.assert_clean()
