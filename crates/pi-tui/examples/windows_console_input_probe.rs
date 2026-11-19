//! Bounded private-console calibration for the Windows native input poll path.
//!
//! Run with `cargo run -p pi-tui --example windows_console_input_probe`.
//! The parent process starts a separate child with CREATE_NEW_CONSOLE. Only
//! that private console receives synthetic records; the caller's console is
//! never attached, read, or modified.

#[cfg(windows)]
mod windows_probe {
    use pi_tui::terminal::{ProcessTerminal, Terminal, TerminalStopOptions};
    use serde::Serialize;
    use std::io::Write;
    use std::os::windows::process::CommandExt;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    use windows_sys::Win32::Foundation::{
        CloseHandle, FILETIME, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
    };
    use windows_sys::Win32::System::Console::{
        FlushConsoleInputBuffer, GetConsoleMode, GetConsoleScreenBufferInfo,
        GetNumberOfConsoleInputEvents, ReadConsoleInputW, SetStdHandle, WriteConsoleInputW,
        CONSOLE_SCREEN_BUFFER_INFO, FOCUS_EVENT, INPUT_RECORD, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentThread, GetProcessIoCounters, GetProcessTimes, GetThreadTimes,
        CREATE_NEW_CONSOLE, IO_COUNTERS,
    };

    const DIRECT_BATCH: usize = 4_096;
    const CONTROL_BATCH: usize = 5_000;
    const IDLE_ITERATIONS: usize = 100;
    const SATURATED_RECORDS: usize = 60_000;
    const PRODUCER_BATCH: usize = 256;

    #[derive(Clone, Copy, Debug, Default, Serialize)]
    struct Measurement {
        read_ops: u64,
        read_bytes: u64,
        write_ops: u64,
        write_bytes: u64,
        other_ops: u64,
        other_bytes: u64,
        process_user_ms: f64,
        process_kernel_ms: f64,
        thread_user_ms: f64,
        thread_kernel_ms: f64,
        wall_ms: f64,
    }

    #[derive(Clone, Copy, Default)]
    struct Sample {
        io: IO_COUNTERS,
        process_user: u64,
        process_kernel: u64,
        thread_user: u64,
        thread_kernel: u64,
    }

    #[derive(Serialize)]
    struct ApiMeasurement {
        case: &'static str,
        requested: usize,
        completed: usize,
        measurement: Measurement,
    }

    #[derive(Serialize)]
    struct PollMeasurement {
        case: &'static str,
        iterations: usize,
        paints: usize,
        records_produced: usize,
        records_remaining: u32,
        measurement: Measurement,
    }

    #[derive(Serialize)]
    struct Report {
        format: u32,
        executable: String,
        direct_api: Vec<ApiMeasurement>,
        idle_poll: PollMeasurement,
        saturated_poll: PollMeasurement,
    }

    fn wide_null(value: &str) -> Vec<u16> {
        value.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn open_private_console_file(name: &str) -> HANDLE {
        let name = wide_null(name);
        unsafe {
            CreateFileW(
                name.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            )
        }
    }

    fn valid(handle: HANDLE) -> bool {
        !handle.is_null() && handle != INVALID_HANDLE_VALUE
    }

    fn filetime_value(value: FILETIME) -> u64 {
        (u64::from(value.dwHighDateTime) << 32) | u64::from(value.dwLowDateTime)
    }

    fn sample() -> Sample {
        unsafe {
            let mut io = IO_COUNTERS::default();
            assert_ne!(GetProcessIoCounters(GetCurrentProcess(), &mut io), 0);
            let mut creation = FILETIME::default();
            let mut exit = FILETIME::default();
            let mut kernel = FILETIME::default();
            let mut user = FILETIME::default();
            assert_ne!(
                GetProcessTimes(
                    GetCurrentProcess(),
                    &mut creation,
                    &mut exit,
                    &mut kernel,
                    &mut user,
                ),
                0
            );
            let process_user = filetime_value(user);
            let process_kernel = filetime_value(kernel);
            let mut thread_creation = FILETIME::default();
            let mut thread_exit = FILETIME::default();
            assert_ne!(
                GetThreadTimes(
                    GetCurrentThread(),
                    &mut thread_creation,
                    &mut thread_exit,
                    &mut kernel,
                    &mut user,
                ),
                0
            );
            Sample {
                io,
                process_user,
                process_kernel,
                thread_user: filetime_value(user),
                thread_kernel: filetime_value(kernel),
            }
        }
    }

    fn ticks_ms(ticks: u64) -> f64 {
        ticks as f64 / 10_000.0
    }

    fn measure<T>(body: impl FnOnce() -> T) -> (T, Measurement) {
        let before = sample();
        let started = std::time::Instant::now();
        let result = body();
        let after = sample();
        let delta = |after_value: u64, before_value: u64| after_value.saturating_sub(before_value);
        (
            result,
            Measurement {
                read_ops: delta(after.io.ReadOperationCount, before.io.ReadOperationCount),
                read_bytes: delta(after.io.ReadTransferCount, before.io.ReadTransferCount),
                write_ops: delta(after.io.WriteOperationCount, before.io.WriteOperationCount),
                write_bytes: delta(after.io.WriteTransferCount, before.io.WriteTransferCount),
                other_ops: delta(after.io.OtherOperationCount, before.io.OtherOperationCount),
                other_bytes: delta(after.io.OtherTransferCount, before.io.OtherTransferCount),
                process_user_ms: ticks_ms(delta(after.process_user, before.process_user)),
                process_kernel_ms: ticks_ms(delta(after.process_kernel, before.process_kernel)),
                thread_user_ms: ticks_ms(delta(after.thread_user, before.thread_user)),
                thread_kernel_ms: ticks_ms(delta(after.thread_kernel, before.thread_kernel)),
                wall_ms: started.elapsed().as_secs_f64() * 1_000.0,
            },
        )
    }

    fn pending_records(input: HANDLE) -> u32 {
        let mut count = 0;
        assert_ne!(
            unsafe { GetNumberOfConsoleInputEvents(input, &mut count) },
            0
        );
        count
    }

    fn drain_lifecycle_records(input: HANDLE) {
        for _ in 0..16 {
            let remaining = pending_records(input);
            if remaining == 0 {
                break;
            }
            assert!(remaining <= 64, "unexpected private lifecycle backlog");
            for _ in 0..remaining {
                let mut record: INPUT_RECORD = unsafe { std::mem::zeroed() };
                let mut read = 0;
                assert_ne!(
                    unsafe { ReadConsoleInputW(input, &mut record, 1, &mut read) },
                    0
                );
                assert_eq!(read, 1);
                assert_eq!(
                    record.EventType, FOCUS_EVENT as u16,
                    "unexpected lifecycle record type"
                );
            }
        }
        assert_eq!(
            pending_records(input),
            0,
            "private lifecycle records did not settle"
        );
    }

    fn write_records(input: HANDLE, records: &[INPUT_RECORD]) -> usize {
        let mut written = 0;
        assert_ne!(
            unsafe {
                WriteConsoleInputW(input, records.as_ptr(), records.len() as u32, &mut written)
            },
            0,
            "WriteConsoleInputW failed: {}",
            std::io::Error::last_os_error()
        );
        written as usize
    }

    fn synthetic_records(count: usize) -> Vec<INPUT_RECORD> {
        (0..count)
            .map(|_| {
                let mut record: INPUT_RECORD = unsafe { std::mem::zeroed() };
                record.EventType = FOCUS_EVENT as u16;
                record
            })
            .collect()
    }

    fn direct_api(input: HANDLE, output: HANDLE) -> Vec<ApiMeasurement> {
        drain_lifecycle_records(input);
        let mut measurements = Vec::new();
        for (case, body) in [
            (
                "get_number_count",
                Box::new(move || {
                    let mut completed = 0usize;
                    for _ in 0..CONTROL_BATCH {
                        pending_records(input);
                        completed += 1;
                    }
                    completed
                }) as Box<dyn FnOnce() -> usize>,
            ),
            (
                "get_console_mode",
                Box::new(move || {
                    let mut completed = 0usize;
                    for _ in 0..CONTROL_BATCH {
                        let mut mode = 0;
                        assert_ne!(unsafe { GetConsoleMode(input, &mut mode) }, 0);
                        completed += 1;
                    }
                    completed
                }),
            ),
            (
                "get_screen_buffer_info",
                Box::new(move || {
                    let mut completed = 0usize;
                    for _ in 0..CONTROL_BATCH {
                        let mut info: CONSOLE_SCREEN_BUFFER_INFO = unsafe { std::mem::zeroed() };
                        assert_ne!(unsafe { GetConsoleScreenBufferInfo(output, &mut info) }, 0);
                        completed += 1;
                    }
                    completed
                }),
            ),
        ] {
            let requested = CONTROL_BATCH;
            let (completed, measurement) = measure(body);
            measurements.push(ApiMeasurement {
                case,
                requested,
                completed,
                measurement,
            });
        }

        assert_ne!(unsafe { FlushConsoleInputBuffer(input) }, 0);
        assert_eq!(
            write_records(input, &synthetic_records(DIRECT_BATCH)),
            DIRECT_BATCH
        );
        assert_eq!(pending_records(input), DIRECT_BATCH as u32);
        let (completed, measurement) = measure(|| {
            let mut completed = 0usize;
            for _ in 0..DIRECT_BATCH {
                let mut remaining = 0;
                let mut record: INPUT_RECORD = unsafe { std::mem::zeroed() };
                unsafe {
                    assert_ne!(GetNumberOfConsoleInputEvents(input, &mut remaining), 0);
                    assert!(remaining > 0);
                    let mut read = 0;
                    assert_ne!(ReadConsoleInputW(input, &mut record, 1, &mut read), 0);
                    assert_eq!(read, 1);
                }
                completed += 1;
            }
            completed
        });
        measurements.push(ApiMeasurement {
            case: "get_number_plus_read_one",
            requested: DIRECT_BATCH,
            completed,
            measurement,
        });
        drain_lifecycle_records(input);
        measurements
    }

    fn poll_loop<ShouldStop>(
        terminal: &mut ProcessTerminal,
        case: &'static str,
        records_produced: usize,
        mut should_stop: ShouldStop,
    ) -> PollMeasurement
    where
        ShouldStop: FnMut(usize) -> bool,
    {
        let input = unsafe { windows_sys::Win32::System::Console::GetStdHandle(STD_INPUT_HANDLE) };
        let mut iterations = 0;
        let mut paints = 0;
        let ((), measurement) = measure(|| {
            let mut last_paint = Instant::now();
            while !should_stop(iterations) {
                assert!(terminal.poll_input().expect("production input poll failed"));
                iterations += 1;
                if last_paint.elapsed() >= Duration::from_millis(80) {
                    terminal.write("private probe paint\n");
                    paints += 1;
                    last_paint = Instant::now();
                }
                std::thread::sleep(Duration::from_millis(16));
            }
        });
        PollMeasurement {
            case,
            iterations,
            paints,
            records_produced,
            records_remaining: pending_records(input),
            measurement,
        }
    }

    fn stop_terminal(terminal: &mut ProcessTerminal) {
        terminal.stop(TerminalStopOptions::default());
    }

    fn producer(count: usize) {
        let input = open_private_console_file("CONIN$");
        assert!(valid(input), "producer CONIN$ open failed");
        let records = synthetic_records(PRODUCER_BATCH);
        let mut produced = 0;
        while produced < count {
            let batch = (count - produced).min(records.len());
            produced += write_records(input, &records[..batch]);
        }
        assert_eq!(produced, count);
        unsafe { CloseHandle(input) };
    }

    fn reader(executable: &str) -> String {
        let input = open_private_console_file("CONIN$");
        let output = open_private_console_file("CONOUT$");
        assert!(valid(input), "reader CONIN$ open failed");
        assert!(valid(output), "reader CONOUT$ open failed");
        unsafe {
            assert_ne!(SetStdHandle(STD_INPUT_HANDLE, input), 0);
            assert_ne!(SetStdHandle(STD_OUTPUT_HANDLE, output), 0);
        }
        // New-conhost startup can insert a focus record asynchronously. It is
        // private fixture noise, not part of any measured batch.
        std::thread::sleep(Duration::from_millis(100));
        assert_ne!(unsafe { FlushConsoleInputBuffer(input) }, 0);
        assert_eq!(pending_records(input), 0);

        let direct_api = direct_api(input, output);
        let mut idle_terminal = ProcessTerminal::new();
        idle_terminal.start(Box::new(|_| {}), Box::new(|| {}));
        let idle_poll = poll_loop(
            &mut idle_terminal,
            "idle_16ms_poll_80ms_paint",
            0,
            |iterations| iterations >= IDLE_ITERATIONS,
        );
        stop_terminal(&mut idle_terminal);

        assert_ne!(unsafe { FlushConsoleInputBuffer(input) }, 0);
        let mut producer_process = Command::new(executable)
            .arg("--producer")
            .arg(SATURATED_RECORDS.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("private console producer spawn failed");
        let mut saturated_terminal = ProcessTerminal::new();
        saturated_terminal.start(Box::new(|_| {}), Box::new(|| {}));
        let saturated_poll = {
            let mut previous_remaining = usize::MAX;
            let mut stalled_iterations = 0usize;
            poll_loop(
                &mut saturated_terminal,
                "saturated_focus_records_60000",
                SATURATED_RECORDS,
                |_| {
                    let remaining = pending_records(input) as usize;
                    if remaining == previous_remaining {
                        stalled_iterations += 1;
                    } else {
                        stalled_iterations = 0;
                    }
                    previous_remaining = remaining;
                    assert!(
                        stalled_iterations < 500,
                        "private input drain made no progress"
                    );
                    if remaining != 0 {
                        return false;
                    }
                    match producer_process
                        .try_wait()
                        .expect("private producer status failed")
                    {
                        Some(status) => {
                            assert!(
                                status.success(),
                                "private producer failed while input was pending"
                            );
                            true
                        }
                        None => false,
                    }
                },
            )
        };
        stop_terminal(&mut saturated_terminal);
        let producer_status = producer_process
            .wait()
            .expect("private producer wait failed");
        assert!(
            producer_status.success(),
            "private producer failed: {producer_status}"
        );
        assert_eq!(saturated_poll.records_remaining, 0);
        unsafe {
            CloseHandle(input);
            CloseHandle(output);
        }
        serde_json::to_string(&Report {
            format: 1,
            executable: executable.to_string(),
            direct_api,
            idle_poll,
            saturated_poll,
        })
        .expect("serialize probe result")
    }

    pub fn run() -> Result<(), String> {
        let executable = std::env::current_exe().map_err(|error| error.to_string())?;
        let args: Vec<String> = std::env::args().collect();
        match args.get(1).map(String::as_str) {
            Some("--reader") => {
                let report = reader(&executable.to_string_lossy());
                let mut stderr = std::io::stderr().lock();
                writeln!(stderr, "{report}").map_err(|error| error.to_string())?;
                stderr.flush().map_err(|error| error.to_string())?;
                Ok(())
            }
            Some("--producer") => {
                let count = args
                    .get(2)
                    .and_then(|value| value.parse().ok())
                    .expect("producer record count");
                producer(count);
                Ok(())
            }
            None => {
                let output = Command::new(&executable)
                    .arg("--reader")
                    .creation_flags(CREATE_NEW_CONSOLE)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::piped())
                    .output()
                    .map_err(|error| format!("private console reader spawn failed: {error}"))?;
                if !output.status.success() {
                    return Err(format!(
                        "private console reader failed: {}: {}",
                        output.status,
                        String::from_utf8_lossy(&output.stderr)
                    ));
                }
                let stderr = String::from_utf8(output.stderr).map_err(|error| error.to_string())?;
                let json = stderr
                    .lines()
                    .find(|line| line.starts_with("{\"format\":1"))
                    .ok_or_else(|| format!("probe JSON missing: {stderr}"))?;
                println!("{json}");
                Ok(())
            }
            Some(other) => Err(format!("unknown private probe mode: {other}")),
        }
    }
}

#[cfg(windows)]
fn main() -> Result<(), String> {
    windows_probe::run()
}

#[cfg(not(windows))]
fn main() {
    println!("windows_console_input_probe is Windows-only");
}
