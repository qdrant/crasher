use anyhow::Context as _;
use futures::{FutureExt as _, TryStreamExt as _, future::BoxFuture};
use std::path::PathBuf;
use tokio::fs;

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
