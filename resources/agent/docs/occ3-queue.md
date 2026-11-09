# OCC-3 sender queue

The status bar shows a blue round `QUEUED n` badge for handoffs waiting to
capture or send (`pending` and `armed`). It alternates filled/bold and
hollow/dim every 500 ms. Published, admitted, completed, failed and cancelled
handoffs do not count. At zero, the badge and its animation deadline disappear.
On narrow terminals, the optional queue badge drops before compacting health or dropping delivery indicators.
The separate OCC3 delivery indicator pulses only on completed native use.

Queue notices are durable audit history, not commands to repeat a handoff.
Only the current pending head can request a fresh capture. Admission clears
source-busy retry scheduling; delayed notices and duplicate admission callbacks
do not re-arm it. A proven receiver-busy refusal before native use remains a
separate bounded retry path. Admission alone is not proof of completed use.

The host serializes timer and event notice drains and checks current durable
acknowledgments before retrying delivery. Sender transcript replay is deduplicated
by notice ID. The parent continues to receive durable notices and coalesced
outcome notifications.

Idle receiver outcomes use the same keyed head-prompt gate as sender events.
The bounded idle timer requests one fresh capture or fallback when needed;
admitted or completed heads never receive a stale retry prompt.

## Capture capacity and completed use

An armed capture holds process-wide capture capacity through publication, not
through the sender session's lifetime. Terminal capture outcomes release that
capacity. Busy retry backoff does not hold it. A fresh capture waits at most
90 seconds for capacity (operator-tunable through `OCC3_CAPTURE_CAPACITY_WAIT_SECONDS`,
clamped to 5-240 seconds) and supports cancellation; refusal does not arm, send,
or replay a capture. Disabling OCC-3 before dispatch returns an armed queue
head to pending without spending a capture attempt. Re-enabling permits an
explicit fresh arm of that same head.

Receiver turn completion forwards the completed-use receipt digest to the
sender. A verified consumed receipt completes the queue entry and lets the next
head proceed. A consumed report without a usable digest remains `uncertain`;
it must not trigger an ordinary-message copy of state already consumed.

## Runtime restart recovery

Closing one session does not evict the shared runtime connector. An authority
refusal can trigger a bounded identity probe. Only a verified change to the
runtime epoch, generation, or model identity evicts that connector's cached
readiness. Other cached connectors remain intact.

The operation that detects a restart still declines once. The next explicit
activation verifies a new connector. Neither the failed capture nor its
publication is replayed automatically. Existing admitted state from the old
runtime does not become valid again through reactivation.
