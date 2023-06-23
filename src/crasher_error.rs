use anyhow::Error;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CrasherError {
    #[error("Run cancelled")]
    Cancelled,
    #[error("Client error - {0:#}")]
    // https://docs.rs/anyhow/latest/anyhow/struct.Error.html#display-representations
    Client(Error),
    #[error("Invariant error - {0}")]
    Invariant(String),
}

impl From<anyhow::Error> for CrasherError {
    fn from(e: anyhow::Error) -> Self {
        CrasherError::Client(e)
    }
}
