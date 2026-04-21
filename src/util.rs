use anyhow::Context as _;
use futures::{FutureExt as _, TryStreamExt as _, future::BoxFuture};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::path::{Path, PathBuf};
use tokio::fs;

pub fn create_rngs(requested_seed: Option<u64>) -> (u64, SmallRng, SmallRng) {
    let s = requested_seed.unwrap_or_else(|| rand::rng().random::<u64>());
    (s, SmallRng::seed_from_u64(s), SmallRng::seed_from_u64(s))
}

/// Per-category size breakdown of a Qdrant `storage/` directory.
///
/// Each file is bucketed into exactly one category (buckets are disjoint;
/// their sum equals `total`):
/// * `dense`             -- `vector_storage-dense-*` without quantization
/// * `dense_quant`       -- `vector_storage-dense-*` using bq/pq/sq/tq/hnsw-inline
/// * `multi_dense`       -- `vector_storage-multi-dense-*` without quantization
/// * `multi_dense_quant` -- `vector_storage-multi-dense-*` using bq/pq/sq/tq
/// * `sparse`            -- `vector_storage-sparse-*`
/// * `payload`           -- `payload_index/` or `payload_storage/` subtrees
/// * `other`             -- segment metadata, WAL, id_tracker, etc.
#[derive(Default, Debug, Clone, Copy)]
pub struct StorageBreakdown {
    pub total: u64,
    pub dense: u64,
    pub dense_quant: u64,
    pub multi_dense: u64,
    pub multi_dense_quant: u64,
    pub sparse: u64,
    pub payload: u64,
    pub other: u64,
}

/// Storage size report: overall totals plus per-shard breakdown.
/// Shards are detected by their position in the tree
/// (`storage/collections/<name>/<shard_id>/...`).
#[derive(Default, Debug)]
pub struct StorageReport {
    pub total: StorageBreakdown,
    pub per_shard: std::collections::BTreeMap<String, StorageBreakdown>,
}

pub async fn storage_report(root: &Path) -> StorageReport {
    let mut report = StorageReport::default();
    // Stack entries: (dir, inherited category tag, inherited shard id)
    let mut stack: Vec<(PathBuf, &'static str, Option<String>)> =
        vec![(root.to_path_buf(), "other", None)];
    while let Some((dir, inherited_tag, shard)) = stack.pop() {
        let mut entries = match fs::read_dir(&dir).await {
            Ok(e) => e,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().into_owned();
            let child_path = entry.path();
            let ft = match entry.file_type().await {
                Ok(t) => t,
                Err(_) => continue,
            };
            if ft.is_dir() {
                let next_tag = classify(&name, inherited_tag);
                let next_shard = if shard.is_some() {
                    shard.clone()
                } else if is_shard_dir(&child_path) {
                    Some(name.clone())
                } else {
                    None
                };
                stack.push((child_path, next_tag, next_shard));
            } else if let Ok(md) = entry.metadata().await {
                let len = md.len();
                add_bucket(&mut report.total, inherited_tag, len);
                if let Some(sid) = &shard {
                    add_bucket(
                        report.per_shard.entry(sid.clone()).or_default(),
                        inherited_tag,
                        len,
                    );
                }
            }
        }
    }
    report
}

fn classify(dir_name: &str, inherited: &'static str) -> &'static str {
    // Once we're inside a categorized subtree, stay in it.
    if inherited != "other" {
        return inherited;
    }
    if let Some(rest) = dir_name.strip_prefix("vector_storage-") {
        // Detect quantized variants by their name tokens. The crasher config
        // uses `-bq-`, `-pq-`, `-sq-`, `-tq-` as the quantization tag, and
        // `-hnsw-inline-` which applies BQ under the hood.
        let quant = rest.contains("-bq-")
            || rest.contains("-pq-")
            || rest.contains("-sq-")
            || rest.contains("-tq-")
            || rest.contains("-hnsw-inline-");
        if rest.starts_with("multi-dense-") {
            return if quant {
                "multi_dense_quant"
            } else {
                "multi_dense"
            };
        }
        if rest.starts_with("sparse-") {
            return "sparse";
        }
        return if quant { "dense_quant" } else { "dense" };
    }
    if dir_name == "payload_index" || dir_name == "payload_storage" {
        return "payload";
    }
    "other"
}

/// A directory is a shard dir if its grandparent is named `collections`,
/// i.e. the path looks like `.../collections/<collection_name>/<shard_id>`.
fn is_shard_dir(path: &Path) -> bool {
    path.parent()
        .and_then(|p| p.parent())
        .and_then(|pp| pp.file_name())
        .map(|n| n == "collections")
        .unwrap_or(false)
}

fn add_bucket(bd: &mut StorageBreakdown, tag: &str, len: u64) {
    bd.total += len;
    match tag {
        "dense" => bd.dense += len,
        "dense_quant" => bd.dense_quant += len,
        "multi_dense" => bd.multi_dense += len,
        "multi_dense_quant" => bd.multi_dense_quant += len,
        "sparse" => bd.sparse += len,
        "payload" => bd.payload += len,
        _ => bd.other += len,
    }
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
