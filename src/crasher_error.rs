use anyhow::Error;
use qdrant_client::QdrantError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CrasherError {
    #[error("Run cancelled")]
    Cancelled,
    #[error("Client error - {0:#}")]
    // https://docs.rs/anyhow/latest/anyhow/struct.Error.html#display-representations
    Client(Error), // assume it can be retried
    #[error("Invariant error - {0}")]
    Invariant(String), // invalid state detected - end of execution
}

impl From<anyhow::Error> for CrasherError {
    fn from(e: anyhow::Error) -> Self {
        Self::Client(e)
    }
}

impl From<reqwest::Error> for CrasherError {
    fn from(e: reqwest::Error) -> Self {
        Self::Client(e.into())
    }
}

impl From<serde_json::Error> for CrasherError {
    fn from(e: serde_json::Error) -> Self {
        Self::Client(e.into())
    }
}

impl From<QdrantError> for CrasherError {
    fn from(err: QdrantError) -> Self {
        // Network error and timeout are detected as transient errors
        match &err {
            QdrantError::Io(_) => anyhow::anyhow!(err).into(),
            QdrantError::ResponseError { status } => {
                if status.code() == tonic::Code::NotFound || status.code() == tonic::Code::Cancelled
                {
                    Self::Invariant(format!("{err}"))
                } else {
                    anyhow::anyhow!(err).into()
                }
            }
            QdrantError::ResourceExhaustedError {
                status: _,
                retry_after_seconds: _,
            } => anyhow::anyhow!(err).into(),
            QdrantError::PayloadDeserialization(_) => Self::Invariant(format!("{err}")),
            QdrantError::ConversionError(_) => Self::Invariant(format!("{err}")),
            QdrantError::InvalidUri(_) => Self::Invariant(format!("{err}")),
            QdrantError::NoSnapshotFound(_) => Self::Invariant(format!("{err}")),
            QdrantError::Reqwest(_) => Self::Invariant(format!("{err}")),
            QdrantError::JsonToPayload(_) => Self::Invariant(format!("{err}")),
        }
    }
}
