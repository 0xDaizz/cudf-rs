//! Error types for cudf operations.
//!
//! All fallible operations return [`Result<T>`], which uses [`CudfError`]
//! as the error type. C++ exceptions from libcudf are automatically
//! converted to `CudfError::Cxx` variants.

/// Errors that can occur during cudf operations.
#[derive(Debug, thiserror::Error)]
pub enum CudfError {
    /// An error originating from the libcudf C++ library.
    #[error("cudf error: {0}")]
    Cxx(String),

    /// A CUDA runtime error.
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// Invalid argument passed to a cudf function.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Type mismatch between expected and actual column types.
    #[error("type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        expected: String,
        actual: String,
    },

    /// Index out of bounds.
    #[error("index out of bounds: {index} (size: {size})")]
    IndexOutOfBounds {
        index: usize,
        size: usize,
    },

    /// An I/O error (e.g., file not found when reading parquet).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// An error from the Arrow interop layer.
    #[cfg(feature = "arrow-interop")]
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
}

impl CudfError {
    /// Convert a cxx::Exception into a CudfError.
    ///
    /// Attempts to classify the exception based on its message:
    /// - Messages containing "CUDA" -> `CudfError::Cuda`
    /// - Everything else -> `CudfError::Cxx`
    pub(crate) fn from_cxx(e: cxx::Exception) -> Self {
        let msg = e.what().to_string();
        if msg.contains("CUDA") || msg.contains("cuda") {
            Self::Cuda(msg)
        } else {
            Self::Cxx(msg)
        }
    }
}

/// Result type alias using [`CudfError`].
pub type Result<T> = std::result::Result<T, CudfError>;
