# Native worker startup adoption

A live worker with temporarily unavailable process-start identity remains registered
as recovering. The supervisor does not invent an identity, signal the process,
replay prompts, or launch replacement work. A failed initial authentication or
subscription probe uses the same deferred recovery ladder as a lost connection.

Healthy workers still adopt synchronously. Deferred probes use the existing bounded
retry delays and ten-round limit. Explicit stop requests, supervisor ownership, and
registration identity remain authoritative. A positively absent or replaced process
still follows the existing dead-worker recovery policy. Failed reconnect cleanup
removes only its own client, never a newer client.

Offline tests use isolated ownership registries, synthetic authentication tokens,
scripted local sockets, and deterministic identity observations. No provider or
operator session is used. Daemon commands and protocol schemas are unchanged.
