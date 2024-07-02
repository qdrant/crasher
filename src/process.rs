use crate::args::Args;
use crate::client::wait_server_ready;
use crate::util;
use anyhow::Context as _;
use qdrant_client::Qdrant;
use rand::Rng;
use std::io;
use std::process::exit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::fs;
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
    pub backup_working_dir: Option<String>,
    pub binary_path: String,
    pub child_process: Child,
}

impl ProcessManager {
    pub fn from_args(args: &Args) -> io::Result<Self> {
        let mut manager = Self::new(&args.working_dir, &args.exec_path)?;

        if let Some(backup_working_dir) = &args.backup_working_dir {
            manager.with_backup_working_dir(backup_working_dir);
        }

        Ok(manager)
    }

    pub fn new(working_dir: &str, binary_path: &str) -> io::Result<Self> {
        let child = start_process(working_dir, binary_path)?;

        Ok(Self {
            working_dir: working_dir.to_string(),
            backup_working_dir: None,
            binary_path: binary_path.to_string(),
            child_process: child,
        })
    }

    pub fn with_backup_working_dir(&mut self, backup_working_dir: &str) -> &mut Self {
        self.backup_working_dir = Some(backup_working_dir.into());
        self
    }

    // Doc says: This is equivalent to sending a SIGKILL on unix platforms.
    pub async fn kill_process(&mut self) {
        self.child_process.kill().await.unwrap();
    }

    pub async fn backup_working_dir(&self) -> anyhow::Result<()> {
        let Some(backup) = &self.backup_working_dir else {
            return Ok(());
        };

        let backup_exists = fs::try_exists(backup)
            .await
            .with_context(|| format!("failed to query if backup working dir {backup} exists"))?;

        if backup_exists {
            fs::remove_dir_all(backup)
                .await
                .with_context(|| format!("failed to remove backup working dir {backup}"))?;
        }

        util::copy_dir(&self.working_dir, backup).await?;

        Ok(())
    }

    pub async fn chaos(
        &mut self,
        stopped: Arc<AtomicBool>,
        client: &Qdrant,
        crash_probability: f64,
        sleep_duration_between_crash_sec: usize,
    ) {
        loop {
            if stopped.load(Ordering::Relaxed) {
                break;
            }

            let drawn = {
                let mut rng = rand::thread_rng();
                rng.gen_bool(crash_probability)
            };
            if drawn {
                log::info!("** Restarting qdrant **");
                self.kill_process().await;

                if let Err(err) = self.backup_working_dir().await {
                    log::error!(
                        "Failed to backup working dir {} to {}: {err}",
                        self.working_dir,
                        self.backup_working_dir
                            .as_ref()
                            .expect("backup working dir"),
                    );
                }

                self.child_process = start_process(&self.working_dir, &self.binary_path).unwrap();

                if let Err(e) = wait_server_ready(client, stopped.clone()).await {
                    log::error!("Failed to wait for qdrant to be ready: {}", e);
                    exit(1);
                }

                log::info!("Qdrant is ready!");
            }
            // wait a bit before next chaos
            sleep(Duration::from_secs(sleep_duration_between_crash_sec as u64)).await;
        }
    }
}
