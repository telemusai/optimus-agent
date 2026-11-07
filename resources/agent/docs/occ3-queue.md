# OCC-3 sender queue

The status bar shows a blue round `QUEUED n` badge for handoffs waiting to
capture or send (`pending` and `armed`). It alternates filled/bold and
hollow/dim every 500 ms. Published, admitted, completed, failed and cancelled
handoffs do not count. At zero, the badge and its animation deadline disappear.
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
