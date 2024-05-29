mod args;
mod client;
mod crasher_error;
mod generators;
mod process;
mod workload;

use crate::args::Args;
use crate::client::wait_server_ready;
use crate::process::ProcessManager;
use crate::workload::Workload;
use clap::Parser;
use env_logger::Target;
use qdrant_client::client::{QdrantClient, QdrantClientConfig};
use std::process::exit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() {
    let args = Args::parse();
    setup_logger();

    let stopped = Arc::new(AtomicBool::new(false));
    let r = stopped.clone();

    ctrlc::set_handler(move || {
        log::info!("Crasher is stopping");
        r.store(true, Ordering::Relaxed);
    })
    .expect("Error setting Ctrl-C handler");

    let client_config = get_config(args.uris.first().unwrap(), args.grpc_timeout_ms);
    let client = QdrantClient::new(Some(client_config)).unwrap();
    let client = Arc::new(client).clone();
    let args = Arc::new(args);
    let crash_probability = args.crash_probability;
    let sleep_duration_between_crash_sec = args.sleep_duration_between_crash_sec;

    match ProcessManager::new(&args.working_dir, &args.exec_path) {
        Err(e) => {
            log::error!("Failed to start Qdrant: {}", e);
        }
        Ok(mut process_manager) => {
            match process_manager.child_process.id() {
                Some(child_process_id) => {
                    log::info!("Child qdrant process id {:?}", child_process_id);
                    log::info!("Waiting for qdrant to be ready...");
                    if let Err(e) = wait_server_ready(&client.clone(), stopped.clone()).await {
                        log::error!("Failed to wait for qdrant to be ready: {}", e);
                        exit(1)
                    }
                    log::info!(
                        "Qdrant is ready! Crashing it with a probability of {}%",
                        crash_probability
                    );

                    // workload task
                    let client_worker = client.clone();
                    let workload = Workload::new(stopped.clone(), args.points_count);
                    let workload_task = tokio::spawn(async move {
                        workload.work(&client_worker, args.clone()).await;
                    });

                    // get started a bit before chaos
                    sleep(Duration::from_secs(10)).await;

                    // process manager task
                    let process_manager_task = tokio::spawn(async move {
                        process_manager
                            .chaos(
                                stopped.clone(),
                                &client.clone(),
                                crash_probability,
                                sleep_duration_between_crash_sec,
                            )
                            .await;
                    });

                    // wait for tasks to finish
                    process_manager_task.await.unwrap();
                    workload_task.await.unwrap();
                }
                None => {
                    log::error!("Failed to get child id");
                }
            }
        }
    }
}

fn get_config(url: &str, timeout_ms: usize) -> QdrantClientConfig {
    let mut config = QdrantClientConfig::from_url(url);
    config.timeout = Duration::from_millis(timeout_ms as u64);
    config.connect_timeout = Duration::from_millis(timeout_ms as u64);
    config
}

pub fn setup_logger() {
    let mut log_builder = env_logger::Builder::new();

    log_builder
        .target(Target::Stdout)
        .format_timestamp_millis()
        .filter_level(log::LevelFilter::Info);

    log_builder.init();
}
