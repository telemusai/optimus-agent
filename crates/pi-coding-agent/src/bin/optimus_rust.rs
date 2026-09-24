//! Native Optimus application entry point.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Offline package inspection must not open profiles, start a daemon, or load providers.
    if args.len() == 2 && args[1] == "--build-provenance" {
        println!("{}", include_str!(concat!(env!("OUT_DIR"), "/build-provenance.json")));
        return;
    }
    // Resolve bundled resources before any runtime threads or child processes start.
    if std::env::var_os("PI_PACKAGE_DIR").filter(|value| !value.is_empty()).is_none() {
        std::env::set_var("PI_PACKAGE_DIR", pi_coding_agent::config::get_package_dir());
    }
    if args.get(1).map(String::as_str) == Some("--internal-telegram-worker") {
        // A standalone worker must not enter the UI or start another daemon.
        let result = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all()
            .build().map_err(|error| error.to_string()).and_then(|runtime| {
                runtime.block_on(pi_coding_agent::modes::telegram::worker::telegram_worker_main(&args))
            });
        if let Err(error) = result { eprintln!("{error}"); std::process::exit(1); }
        return;
    }
    std::process::exit(pi_coding_agent::cli_entry::main_entry(args));
}
