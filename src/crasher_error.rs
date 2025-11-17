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
        CrasherError::Client(e)
    }
}

impl From<reqwest::Error> for CrasherError {
    fn from(e: reqwest::Error) -> Self {
        CrasherError::Client(e.into())
    }
}

impl From<qdrant_client::QdrantError> for CrasherError {
    fn from(err: qdrant_client::QdrantError) -> Self {
        // Network error and timeout are detected as transient errors
        match &err {
            QdrantError::Io(_) => anyhow::anyhow!(err).into(),
            QdrantError::ResponseError { status } => {
                if status.code() == tonic::Code::NotFound || status.code() == tonic::Code::Cancelled
                {
                    CrasherError::Invariant(format!("{err}"))
                } else {
                    anyhow::anyhow!(err).into()
                }
            }
            QdrantError::ResourceExhaustedError {
                status: _,
                retry_after_seconds: _,
            } => anyhow::anyhow!(err).into(),
            QdrantError::PayloadDeserialization(_) => CrasherError::Invariant(format!("{err}")),
            QdrantError::ConversionError(_) => CrasherError::Invariant(format!("{err}")),
            QdrantError::InvalidUri(_) => CrasherError::Invariant(format!("{err}")),
            QdrantError::NoSnapshotFound(_) => CrasherError::Invariant(format!("{err}")),
            QdrantError::Reqwest(_) => CrasherError::Invariant(format!("{err}")),
            QdrantError::JsonToPayload(_) => CrasherError::Invariant(format!("{err}")),
        }
    }
}
