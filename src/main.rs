mod args;
mod checker;
mod client;
mod crasher_error;
mod generators;
mod process;
mod util;
mod workload;

use crate::args::Args;
use crate::client::wait_server_ready;
use crate::process::ProcessManager;
use crate::util::create_rngs;
use crate::workload::Workload;
use clap::Parser;
use env_logger::Target;
use qdrant_client::{Qdrant, config::QdrantConfig};
use reqwest::Client;
use std::process::exit;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::time::sleep;

const COLLECTION_NAME: &str = "workload-crasher";

#[tokio::main]
async fn main() {
    let args = Args::parse();
    setup_logger();

    let stopped = Arc::new(AtomicBool::new(false));
    let r = stopped.clone();

    ctrlc::set_handler(move || {
        log::info!("Stopping Crasher");
        r.store(true, Ordering::Relaxed);
    })
    .expect("Error setting Ctrl-C handler");

    match ProcessManager::from_args(&args) {
        Err(err) => log::error!("Failed to start Qdrant: {err}"),
        Ok(mut process_manager) => {
            match process_manager.child_process.id() {
                Some(child_process_id) => {
                    log::info!("Child qdrant process id {child_process_id:?}");

                    // gRPC client
                    let grpc_client_config =
                        get_grpc_config(args.uris.first().unwrap(), args.grpc_timeout_ms as usize);
                    let grpc_client = Qdrant::new(grpc_client_config).unwrap();

                    // HTTP client
                    let http_client = Client::builder()
                        .timeout(Duration::from_millis(args.http_timeout_ms))
                        .connect_timeout(Duration::from_millis(args.http_timeout_ms))
                        .user_agent("crasher :fire:")
                        .build()
                        .unwrap();

                    let crash_probability = args.crash_probability;
                    let sleep_duration_between_crash_sec = args.sleep_duration_between_crash_sec;

                    log::info!("Waiting for qdrant to be ready...");
                    if let Err(err) = wait_server_ready(&grpc_client, stopped.clone(), false).await
                    {
                        log::error!("Failed to wait for qdrant to be ready: {err:?}");
                        exit(1)
                    }
                    log::info!(
                        "Qdrant is ready! Crashing it with a probability of {}%",
                        crash_probability * 100.0,
                    );

                    let collection_name = COLLECTION_NAME;
                    let crash_lock = Arc::new(tokio::sync::Mutex::new(()));
                    let (rng_seed, mut workload_rng, mut chaos_rng) = create_rngs(args.rng_seed);

                    // workload task
                    let workload: Workload = Workload::new(
                        collection_name,
                        stopped.clone(),
                        crash_lock.clone(),
                        args.duplication_factor,
                        args.points_count,
                        args.vector_dimension,
                        rng_seed,
                    );
                    let args = Arc::new(args);
                    let client_worker = grpc_client.clone();
                    let workload_task = tokio::spawn(async move {
                        workload
                            .work(&client_worker, &http_client, args, &mut workload_rng)
                            .await;
                    });

                    // get started a bit before chaos
                    sleep(Duration::from_secs(10)).await;

                    // process manager task
                    let process_manager_task = tokio::spawn(async move {
                        process_manager
                            .chaos(
                                stopped,
                                crash_lock,
                                &grpc_client,
                                crash_probability as f64,
                                sleep_duration_between_crash_sec,
                                &mut chaos_rng,
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

fn get_grpc_config(url: &str, timeout_ms: usize) -> QdrantConfig {
    Qdrant::from_url(url)
        .timeout(Duration::from_millis(timeout_ms as u64))
        .connect_timeout(Duration::from_millis(timeout_ms as u64))
}

pub fn setup_logger() {
    let mut log_builder = env_logger::Builder::new();

    log_builder
        .target(Target::Stdout)
        .format_timestamp_millis()
        .filter_level(log::LevelFilter::Info);

    log_builder.init();
}
