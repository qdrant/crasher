use crate::client::wait_server_ready;
use qdrant_client::client::QdrantClient;
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::time::sleep;

pub fn start_process(working_dir_path: &str, exec_path: &str) -> tokio::io::Result<Child> {
    Command::new(exec_path)
        .current_dir(working_dir_path)
        //.stdout(std::process::Stdio::piped())
        //.stderr(std::process::Stdio::piped())
        .kill_on_drop(true) // kill child process if parent is dropped
        .spawn()
}

pub struct ProcessManager {
    pub working_dir: String,
    pub binary_path: String,
    pub child_process: Child,
}

impl ProcessManager {
    pub fn new(working_dir: &str, binary_path: &str) -> tokio::io::Result<Self> {
        let child = start_process(working_dir, binary_path)?;
        Ok(Self {
            working_dir: working_dir.to_string(),
            binary_path: binary_path.to_string(),
            child_process: child,
        })
    }

    pub async fn kill_process(&mut self) {
        self.child_process.kill().await.unwrap();
    }

    pub async fn chaos(&mut self, stopped: Arc<AtomicBool>, client: &QdrantClient, crash_probability: f64) {
        loop {
            if stopped.load(Ordering::SeqCst) {
                break;
            }

            let drawn = {
                let mut rng = rand::thread_rng();
                rng.gen_bool(crash_probability)
            };
            if drawn {
                log::info!("** Restarting qdrant **");
                self.kill_process().await;
                self.child_process = start_process(&self.working_dir, &self.binary_path).unwrap();
                if let Err(e) = wait_server_ready(client, stopped.clone()).await {
                    log::error!("Failed to wait for qdrant to be ready: {}", e);
                }
                log::info!("Qdrant is ready!");
            }
            sleep(Duration::from_secs(5)).await;
        }
    }
}
