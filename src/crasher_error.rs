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
                let code = status.code();
                // A `Cancelled` produced locally because the connection dropped (e.g. Qdrant was
                // killed by the crasher) is not a server-side cancellation and can be retried.
                let transport_cancel =
                    code == tonic::Code::Cancelled && is_client_transport_cancel(status);
                // Otherwise `Cancelled` is a server-side cancellation (most likely a deadlock)
                // or a client timeout.
                if !transport_cancel
                    && (code == tonic::Code::NotFound || code == tonic::Code::Cancelled)
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

/// Whether a `Cancelled` status was produced locally by hyper because the connection went away
/// (e.g. Qdrant was killed), rather than sent by the server or caused by a client timeout.
///
/// tonic keeps the original transport error as the status source; a status received from the
/// server has no source, and a timeout carries `tonic::TimeoutExpired` instead.
fn is_client_transport_cancel(status: &tonic::Status) -> bool {
    let mut source = std::error::Error::source(status);
    while let Some(err) = source {
        if let Some(hyper_err) = err.downcast_ref::<hyper::Error>() {
            return hyper_err.is_canceled();
        }
        source = err.source();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real `hyper::Error` with `is_canceled() == true`: send on a connection that is gone.
    async fn hyper_canceled_error() -> hyper::Error {
        let (io, _peer) = tokio::io::duplex(64);
        let (mut sender, conn) =
            hyper::client::conn::http1::handshake::<_, String>(hyper_util::rt::TokioIo::new(io))
                .await
                .unwrap();
        drop(conn);
        let err = sender
            .send_request(hyper::Request::new(String::new()))
            .await
            .unwrap_err();
        assert!(err.is_canceled());
        err
    }

    fn classify(status: tonic::Status) -> CrasherError {
        QdrantError::ResponseError { status }.into()
    }

    #[tokio::test]
    async fn transport_cancel_is_retryable() {
        let status = tonic::Status::from_error(Box::new(hyper_canceled_error().await));
        assert_eq!(status.code(), tonic::Code::Cancelled);
        assert!(matches!(classify(status), CrasherError::Client(_)));
    }

    #[test]
    fn server_cancel_is_invariant() {
        let status = tonic::Status::cancelled("operation was canceled");
        assert!(matches!(classify(status), CrasherError::Invariant(_)));
    }

    #[test]
    fn timeout_is_invariant() {
        let status = tonic::Status::from_error(Box::new(tonic::TimeoutExpired(())));
        assert_eq!(status.code(), tonic::Code::Cancelled);
        assert!(matches!(classify(status), CrasherError::Invariant(_)));
    }

    #[test]
    fn not_found_is_invariant() {
        let status = tonic::Status::not_found("missing");
        assert!(matches!(classify(status), CrasherError::Invariant(_)));
    }
}
