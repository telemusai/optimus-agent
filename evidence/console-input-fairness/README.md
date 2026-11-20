# Windows native input fairness calibration

- Base: `f8086f15c269e184f87b7e5f3b618d021615f3e4`.
- Old production input budget: 4096 records per owner poll.
- Candidate budget: 256 records per owner poll.
- Toolchain: `rustc 1.98.1 (48a229cea 2026-09-01)`, dev profile.
- Command: `cargo run --locked -p pi-tui --example windows_console_input_probe`.

The example starts a private `CREATE_NEW_CONSOLE` reader. A separate producer
process writes 60,000 synthetic `FOCUS_EVENT` records into only that private
console. Process I/O counters therefore exclude producer writes. No running
Optimus process or caller console is attached or modified. Before measurements,
the fixture bounds and drains private conhost lifecycle records, accepting only
`FOCUS_EVENT`. Host loops paint when at least 80 ms of wall time has elapsed,
rather than inferring time from the iteration count. The unit tests check
ordered resumption without timing assumptions.

## Direct API accounting

The old-code direct fixed batch made one `GetNumberOfConsoleInputEvents` and one
single-record `ReadConsoleInputW` for each of 4,096 queued records. It measured
exactly 8,192 `OtherOperationCount` operations and 81,920 `OtherTransferCount`
bytes: two operations and 20 transferred bytes per record. The input-count,
console-mode, and screen-buffer-info batches each measured exactly one `Other`
operation per API call. The count-call case is named `get_number_count` because
asynchronous lifecycle records can be present without changing that count.

## Same-fixture old/new saturated result

The old code drained all 60,000 records in 16 owner iterations and 15 paint
opportunities: 120,190 `Other` operations over 3670.3602 ms (32746.1049/s),
1296.875 ms process-kernel time (0.353337 core over the run).

The 256-record budget drained all 60,000 records in 236 owner iterations and 78 paint
opportunities: 122,045 `Other` operations over 7220.3082 ms
(16903.0181/s), 1218.750 ms process-kernel time (0.168795 core over the run).

Added owner-loop size polls explain the small increase in total operations. The
old and new idle controls both performed 795 `Other` operations in 100 iterations and
19 paints, with zero records left. The fixed case spent more wall time draining
this finite extreme burst because it yielded and slept between batches. This is
a scheduler-fairness comparison, not a claim that total flood latency is lower.

These private synthetic results establish API accounting and scheduler
fairness. They do not identify a producer in a live application and do not
reproduce the live application's measured operation rate.
