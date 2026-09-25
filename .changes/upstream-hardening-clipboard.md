Fixed credential migration cleanup after failed durable writes and settings reads that could overwrite unreadable data.
Fixed auth/settings lock ownership with OS-owned locks, including cancellation and crash release. Stop all older Optimus processes before using this build; older directory-lock writers cannot share the new lock files.
Fixed session repair failure handling and bounded cyclic ancestry traversal while preserving valid transcript entries.
Fixed live worker adoption retry and replacement-client cleanup without signaling uncertain process identities.
Fixed local Linux clipboard result checks and SSH/tmux terminal forwarding; terminal clipboard requests are reported as unconfirmed rather than copied.
Fixed managed rg/fd availability in Python kernel PATH while preserving explicit environment overrides and Jev search guidance.
Added bounded accident guards for supported literal destructive Git and recursive-force rm commands in Python bash(); explicit per-call overrides remain available. These checks are not a shell sandbox.
