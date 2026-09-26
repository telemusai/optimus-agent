#[cfg(unix)]
mod unix {
    use pi_coding_agent::core::tools::bash::{
        BashExecOptions, BashOperations, LocalBashOperations,
    };
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn shell_exit_returns_even_when_background_child_holds_output_open() {
        let dir = tempfile::tempdir().unwrap();
        let output = Arc::new(Mutex::new(Vec::new()));
        let captured = output.clone();
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            LocalBashOperations { shell_path: None }.exec(
                "sleep 30 & echo $! > child.pid; printf 'port 8080\\n'",
                dir.path().to_str().unwrap(),
                BashExecOptions {
                    on_data: Arc::new(move |data| captured.lock().unwrap().extend_from_slice(data)),
                    signal: None,
                    timeout: Some(0.5),
                    env: None,
                },
            ),
        )
        .await;
        let pid: i32 = std::fs::read_to_string(dir.path().join("child.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
        assert_eq!(
            result
                .expect("shell exited without waiting for background pipe EOF")
                .unwrap()
                .exit_code,
            Some(0)
        );
        let output = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        assert!(output.contains("8080"));
    }

    #[tokio::test]
    async fn search_stdin_eof_timeout_and_cancellation_settle() {
        let dir = tempfile::tempdir().unwrap();
        for command in ["grep 8080", "printf 'port 8080\\n' | grep 8080"] {
            tokio::time::timeout(
                Duration::from_secs(2),
                LocalBashOperations { shell_path: None }.exec(
                    command,
                    dir.path().to_str().unwrap(),
                    BashExecOptions {
                        on_data: Arc::new(|_| {}),
                        signal: None,
                        timeout: Some(1.),
                        env: None,
                    },
                ),
            )
            .await
            .unwrap()
            .unwrap();
        }
        for cancel in [false, true] {
            let token = CancellationToken::new();
            if cancel {
                let signal = token.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    signal.cancel();
                });
            }
            let result = tokio::time::timeout(
                Duration::from_secs(4),
                LocalBashOperations { shell_path: None }.exec(
                    "sleep 30 | grep 8080",
                    dir.path().to_str().unwrap(),
                    BashExecOptions {
                        on_data: Arc::new(|_| {}),
                        signal: Some(token),
                        timeout: Some(0.1),
                        env: None,
                    },
                ),
            )
            .await
            .unwrap()
            .unwrap_err();
            assert!(
                result.contains(if cancel { "aborted" } else { "timeout" }),
                "{result}"
            );
        }
    }
    #[tokio::test]
    async fn dropping_the_tool_future_kills_its_owned_process() {
        let dir = tempfile::tempdir().unwrap();
        let result = tokio::time::timeout(
            Duration::from_millis(200),
            LocalBashOperations { shell_path: None }.exec(
                "echo $$ > child.pid; exec sleep 30",
                dir.path().to_str().unwrap(),
                BashExecOptions {
                    on_data: Arc::new(|_| {}),
                    signal: None,
                    timeout: None,
                    env: None,
                },
            ),
        )
        .await;
        assert!(result.is_err());
        let pid: i32 = std::fs::read_to_string(dir.path().join("child.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            while unsafe { libc::kill(pid, 0) } == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("dropped Bash future must kill and reap its child");
    }
}
