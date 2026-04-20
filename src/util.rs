use anyhow::Context as _;
use futures::{FutureExt as _, TryStreamExt as _, future::BoxFuture};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::path::{Path, PathBuf};
use tokio::fs;

pub fn create_rngs(requested_seed: Option<u64>) -> (u64, SmallRng, SmallRng) {
    let s = requested_seed.unwrap_or_else(|| rand::rng().random::<u64>());
    (s, SmallRng::seed_from_u64(s), SmallRng::seed_from_u64(s))
}

/// Recursively sum the sizes of all regular files under `path` in bytes.
/// Errors on individual entries are skipped (best-effort for observability).
pub async fn dir_size_bytes(path: &Path) -> u64 {
    let mut stack = vec![path.to_path_buf()];
    let mut total = 0u64;
    while let Some(p) = stack.pop() {
        let mut entries = match fs::read_dir(&p).await {
            Ok(e) => e,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let ft = match entry.file_type().await {
                Ok(t) => t,
                Err(_) => continue,
            };
            if ft.is_dir() {
                stack.push(entry.path());
            } else if let Ok(md) = entry.metadata().await {
                total += md.len();
            }
        }
    }
    total
}

/// Free bytes on the filesystem hosting `path`, via `statvfs`.
pub fn fs_free_bytes(path: &Path) -> Option<u64> {
    nix::sys::statvfs::statvfs(path)
        .ok()
        .map(|s| s.blocks_available() * s.block_size())
}

pub fn format_mb(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1_048_576.0)
}

pub fn copy_dir(
    src: impl Into<PathBuf>,
    dst: impl Into<PathBuf>,
) -> BoxFuture<'static, anyhow::Result<()>> {
    copy_dir_impl(src.into(), dst.into()).boxed()
}

async fn copy_dir_impl(src: PathBuf, dst: PathBuf) -> anyhow::Result<()> {
    let read_dir = fs::read_dir(&src)
        .await
        .with_context(|| format!("failed to read directory {}", src.display()))?;

    let dst_exists = fs::try_exists(&dst)
        .await
        .with_context(|| format!("failed to query if {} directory exists", dst.display()))?;

    if dst_exists {
        return Err(anyhow::format_err!(
            "{} directory already exists",
            dst.display()
        ));
    }

    fs::create_dir_all(&dst)
        .await
        .with_context(|| format!("failed to create directory {}", dst.display()))?;

    let mut entries = tokio_stream::wrappers::ReadDirStream::new(read_dir).map_err(|err| {
        anyhow::Error::new(err).context(format!(
            "failed to read next entry in {} directory",
            src.display(),
        ))
    });

    while let Some(entry) = entries.try_next().await? {
        let entry_type = entry
            .file_type()
            .await
            .with_context(|| format!("failed to read {} entry type", entry.path().display()))?;

        let dst = dst.join(entry.file_name());

        if entry_type.is_dir() {
            copy_dir(entry.path(), dst).await?;
        } else {
            let entry_path = entry.path();

            fs::copy(&entry_path, &dst).await.with_context(|| {
                format!(
                    "failed to copy {} file to {}",
                    entry_path.display(),
                    dst.display()
                )
            })?;
        }
    }

    Ok(())
}
