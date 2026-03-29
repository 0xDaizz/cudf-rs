//! Arrow interoperability -- convert between cudf and arrow-rs types.
//!
//! Provides IPC-based serialization for [`Column`] and [`Table`], plus
//! direct conversion to/from `arrow::RecordBatch` when the `arrow-interop`
//! feature is enabled.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//!
//! let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
//! let ipc = col.to_arrow_ipc().unwrap();
//! let roundtrip = Column::from_arrow_ipc(&ipc).unwrap();
//! assert_eq!(roundtrip.len(), 3);
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;

impl Column {
    /// Export this column to Arrow IPC format (serialized bytes).
    pub fn to_arrow_ipc(&self) -> Result<Vec<u8>> {
        cudf_cxx::interop::ffi::column_to_arrow_ipc(&self.inner).map_err(CudfError::from_cxx)
    }

    /// Import a column from Arrow IPC format.
    pub fn from_arrow_ipc(data: &[u8]) -> Result<Self> {
        let raw =
            cudf_cxx::interop::ffi::column_from_arrow_ipc(data).map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }
}

impl Table {
    /// Export this table to Arrow IPC format (serialized bytes).
    pub fn to_arrow_ipc(&self) -> Result<Vec<u8>> {
        cudf_cxx::interop::ffi::table_to_arrow_ipc(&self.inner).map_err(CudfError::from_cxx)
    }

    /// Import a table from Arrow IPC format.
    pub fn from_arrow_ipc(data: &[u8]) -> Result<Self> {
        let raw =
            cudf_cxx::interop::ffi::table_from_arrow_ipc(data).map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }
}

/// Convert to/from arrow-rs RecordBatch (requires `arrow-interop` feature).
#[cfg(feature = "arrow-interop")]
mod arrow_conv {
    use arrow::ipc;
    use arrow::record_batch::RecordBatch;

    use crate::error::{CudfError, Result};
    use crate::table::Table;

    impl Table {
        /// Convert this table to an `arrow::RecordBatch`.
        ///
        /// This serializes the GPU data to Arrow IPC format and then
        /// deserializes into an arrow-rs RecordBatch on the host.
        pub fn to_record_batch(&self) -> Result<RecordBatch> {
            let ipc_bytes = self.to_arrow_ipc()?;
            let cursor = std::io::Cursor::new(ipc_bytes);
            let reader =
                ipc::reader::FileReader::try_new(cursor, None).map_err(CudfError::Arrow)?;
            reader
                .into_iter()
                .next()
                .ok_or_else(|| CudfError::InvalidArgument("empty IPC stream".into()))?
                .map_err(CudfError::Arrow)
        }

        /// Create a table from an `arrow::RecordBatch`.
        ///
        /// This serializes the RecordBatch to Arrow IPC format and then
        /// imports it into GPU memory via libcudf.
        pub fn from_record_batch(batch: &RecordBatch) -> Result<Self> {
            let mut buf = Vec::new();
            {
                let mut writer = ipc::writer::FileWriter::try_new(&mut buf, batch.schema_ref())
                    .map_err(CudfError::Arrow)?;
                writer.write(batch).map_err(CudfError::Arrow)?;
                writer.finish().map_err(CudfError::Arrow)?;
            }
            Self::from_arrow_ipc(&buf)
        }
    }
}
