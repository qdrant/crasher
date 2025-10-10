use crate::args::Args;
use crate::client::wait_server_ready;
use crate::util;
use anyhow::Context as _;
use qdrant_client::Qdrant;
use rand::Rng;
use std::collections::VecDeque;
use std::io;
use std::process::exit;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::fs;
use tokio::process::{Child, Command};
use tokio::time::sleep;

pub fn start_process(
    working_dir_path: &str,
    exec_path: &str,
    kill_on_drop: bool,
) -> io::Result<Child> {
    Command::new("systemd-run")
        .arg("--user")
        .arg("--scope")
        .arg("-p")
        .arg("CPUQuota=8%")
        .arg("--")
        .arg(exec_path)
        .current_dir(working_dir_path)
        .kill_on_drop(kill_on_drop)
        .env("QDRANT__LOGGER__ON_DISK__ENABLED", "true")
        .env("QDRANT__LOGGER__ON_DISK__LOG_FILE", "./qdrant.log")
        .env("QDRANT__SERVICE__HARDWARE_REPORTING", "true")
        .env("QDRANT__STORAGE__COLLECTION_STRICT_MODE", "true")
        .env("QDRANT__FEATURE_FLAGS__ALL", "true")
        .env("QDRANT__LOG_LEVEL","TRACE,raft::raft=info,actix_http=info,tonic=info,want=info,mio=info")
        .spawn()
}

pub struct ProcessManager {
    pub working_dir: String,
    pub binary_path: String,
    pub backup_dirs: VecDeque<String>,
    pub child_process: Child,
    pub kill_on_drop: bool,
}

impl ProcessManager {
    pub fn from_args(args: &Args) -> io::Result<Self> {
        let manager = Self::new(&args.working_dir, &args.exec_path, args.shutdown_on_error)?
            .with_backup_dirs(args.backup_working_dir.clone());

        Ok(manager)
    }

    pub fn new(working_dir: &str, binary_path: &str, kill_on_drop: bool) -> io::Result<Self> {
        let child = start_process(working_dir, binary_path, kill_on_drop)?;

        Ok(Self {
            working_dir: working_dir.to_string(),
            binary_path: binary_path.to_string(),
            backup_dirs: VecDeque::new(),
            child_process: child,
            kill_on_drop,
        })
    }

    pub fn with_backup_dirs(mut self, backup_dirs: impl Into<VecDeque<String>>) -> Self {
        self.backup_dirs = backup_dirs.into();
        self
    }

    // Doc says: This is equivalent to sending a SIGKILL on unix platforms.
    pub async fn kill_process(&mut self) {
        self.child_process.kill().await.unwrap();
    }

    pub async fn backup_working_dir(&mut self) -> anyhow::Result<()> {
        let Some(backup_dir) = self.backup_dirs.front() else {
            return Ok(());
        };

        let backup_exists = fs::try_exists(backup_dir).await.with_context(|| {
            format!("failed to query if backup working dir {backup_dir} exists")
        })?;

        if backup_exists {
            fs::remove_dir_all(backup_dir)
                .await
                .with_context(|| format!("failed to remove backup working dir {backup_dir}"))?;
        }

        util::copy_dir(&self.working_dir, backup_dir).await?;

        let backup_dir = self.backup_dirs.pop_front().expect("backup dir");
        self.backup_dirs.push_back(backup_dir);

        Ok(())
    }

    pub async fn chaos(
        &mut self,
        stopped: Arc<AtomicBool>,
        crash_lock: Arc<tokio::sync::Mutex<()>>,
        client: &Qdrant,
        crash_probability: f64,
        sleep_duration_between_crash_sec: usize,
    ) {
        loop {
            if stopped.load(Ordering::Relaxed) {
                break;
            }

            let drawn = {
                let mut rng = rand::rng();
                rng.random_bool(crash_probability)
            };
            if drawn {
                let Ok(_crash_lock_guard) = crash_lock.try_lock() else {
                    // give up draw if crashing is not allowed
                    continue;
                };
                log::info!("** Restarting qdrant **");
                self.kill_process().await;

                if let Err(err) = self.backup_working_dir().await {
                    log::error!(
                        "Failed to backup working dir {} to {}: {err:?}",
                        self.working_dir,
                        self.backup_dirs.front().expect("backup dir"),
                    );
                }

                self.child_process =
                    start_process(&self.working_dir, &self.binary_path, self.kill_on_drop).unwrap();

                if let Err(err) = wait_server_ready(client, stopped.clone(), true).await {
                    log::error!("Failed to wait for qdrant to be ready: {err:?}");
                    exit(1);
                }

                log::info!("Qdrant is ready!");
            }
            // wait a bit before next chaos
            sleep(Duration::from_secs(sleep_duration_between_crash_sec as u64)).await;
        }
    }
}
