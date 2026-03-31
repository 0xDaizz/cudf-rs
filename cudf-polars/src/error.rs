//! Error bridging between cudf and polars.

use cudf::CudfError;
use polars_error::{PolarsResult, polars_err};

/// Convert a cudf::Result into a PolarsResult.
pub fn gpu_result<T>(result: cudf::Result<T>) -> PolarsResult<T> {
    result.map_err(|e| match e {
        CudfError::Cxx(msg) => polars_err!(ComputeError: "GPU error: {}", msg),
        CudfError::Cuda(msg) => polars_err!(ComputeError: "GPU CUDA error: {}", msg),
        CudfError::InvalidArgument(msg) => polars_err!(ComputeError: "GPU invalid arg: {}", msg),
        CudfError::TypeMismatch { expected, actual } => {
            polars_err!(ComputeError: "GPU type mismatch: expected {}, got {}", expected, actual)
        }
        CudfError::IndexOutOfBounds { index, size } => {
            polars_err!(ComputeError: "GPU index {} out of bounds (size {})", index, size)
        }
        CudfError::Io(e) => polars_err!(ComputeError: "GPU I/O error: {}", e),
        _ => polars_err!(ComputeError: "GPU error: {}", e),
    })
}
