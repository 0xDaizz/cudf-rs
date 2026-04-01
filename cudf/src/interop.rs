//! Arrow interoperability, DLPack, and contiguous_split.
//!
//! Provides:
//! - IPC-based serialization for [`Column`] and [`Table`] (legacy path)
//! - Arrow C Data Interface for true zero-copy exchange with arrow-rs
//! - DLPack tensor exchange for interop with PyTorch / CuPy
//! - `pack` / `unpack` / `contiguous_split` for efficient GPU buffer management
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//!
//! // Arrow IPC roundtrip
//! let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
//! let ipc = col.to_arrow_ipc().unwrap();
//! let roundtrip = Column::from_arrow_ipc(&ipc).unwrap();
//! assert_eq!(roundtrip.len(), 3);
//!
//! // Pack / unpack
//! let packed = cudf::PackedTable::pack(&Table::new(vec![col]).unwrap()).unwrap();
//! let unpacked = packed.unpack().unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;
use crate::types::checked_i32;

#[cfg(feature = "arrow-interop")]
use arrow_array::Array as _;

// ── Arrow IPC (legacy) ────────────────────────────────────────

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

// ── Arrow C Data Interface (true zero-copy) ───────────────────

#[cfg(feature = "arrow-interop")]
impl Column {
    /// Export this column to an `arrow::ArrayRef` via the Arrow C Data Interface.
    ///
    /// This is the preferred zero-copy path.  The GPU data is copied to host
    /// memory by libcudf **once** (via `ArrowExportPair`), then imported
    /// directly into arrow-rs without additional serialization.
    pub fn to_arrow_array(&self) -> Result<arrow_array::ArrayRef> {
        let mut pair = cudf_cxx::interop::ffi::column_to_arrow_pair(&self.inner)
            .map_err(CudfError::from_cxx)?;
        let schema_ptr = cudf_cxx::interop::ffi::arrow_pair_schema(pair.pin_mut());
        let array_ptr = cudf_cxx::interop::ffi::arrow_pair_array(pair.pin_mut());

        // SAFETY: schema_ptr and array_ptr are valid heap-allocated
        // ArrowSchema / ArrowArray structs produced by the C++ shim.
        // arrow::ffi takes ownership and will call the release callbacks.
        unsafe {
            let ffi_schema = arrow::ffi::FFI_ArrowSchema::from_raw(
                schema_ptr as *mut arrow::ffi::FFI_ArrowSchema,
            );
            let ffi_array =
                arrow::ffi::FFI_ArrowArray::from_raw(array_ptr as *mut arrow::ffi::FFI_ArrowArray);
            let data = arrow::ffi::from_ffi(ffi_array, &ffi_schema).map_err(CudfError::Arrow)?;
            Ok(arrow_array::make_array(data))
        }
    }

    /// Import a column from an `arrow::ArrayRef` via the Arrow C Data Interface.
    pub fn from_arrow_array(array: &dyn arrow_array::Array) -> Result<Self> {
        let data = array.to_data();
        // Export to C Data Interface structs.
        let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&data).map_err(CudfError::Arrow)?;

        // Leak them to heap so the C++ side can consume them.
        let schema_ptr = Box::into_raw(Box::new(ffi_schema)) as u64;
        let array_ptr = Box::into_raw(Box::new(ffi_array)) as u64;

        let result = cudf_cxx::interop::ffi::column_from_arrow_cdata(schema_ptr, array_ptr);
        if result.is_err() {
            // C++ threw before taking ownership — reclaim and drop to prevent leak.
            unsafe {
                drop(Box::from_raw(
                    schema_ptr as *mut arrow::ffi::FFI_ArrowSchema,
                ));
                drop(Box::from_raw(array_ptr as *mut arrow::ffi::FFI_ArrowArray));
            }
        }
        let raw = result.map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }
}

#[cfg(feature = "arrow-interop")]
impl Table {
    /// Export this table to an `arrow::RecordBatch` via the Arrow C Data Interface.
    ///
    /// Preferred zero-copy path (no IPC serialization overhead).
    /// Uses `ArrowExportPair` to perform a single GPU→host transfer.
    pub fn to_arrow_batch(&self) -> Result<arrow::record_batch::RecordBatch> {
        let mut pair = cudf_cxx::interop::ffi::table_to_arrow_pair(&self.inner)
            .map_err(CudfError::from_cxx)?;
        let schema_ptr = cudf_cxx::interop::ffi::arrow_pair_schema(pair.pin_mut());
        let array_ptr = cudf_cxx::interop::ffi::arrow_pair_array(pair.pin_mut());

        unsafe {
            let ffi_schema = arrow::ffi::FFI_ArrowSchema::from_raw(
                schema_ptr as *mut arrow::ffi::FFI_ArrowSchema,
            );
            let ffi_array =
                arrow::ffi::FFI_ArrowArray::from_raw(array_ptr as *mut arrow::ffi::FFI_ArrowArray);

            // Import as a struct array (which represents a RecordBatch).
            let data = arrow::ffi::from_ffi(ffi_array, &ffi_schema).map_err(CudfError::Arrow)?;
            let struct_array = arrow_array::StructArray::from(data);
            Ok(arrow::record_batch::RecordBatch::from(struct_array))
        }
    }

    /// Import a table from an `arrow::RecordBatch` via the Arrow C Data Interface.
    pub fn from_arrow_batch(batch: &arrow::record_batch::RecordBatch) -> Result<Self> {
        // Build a StructArray from the batch's columns + schema.
        let struct_array = arrow_array::StructArray::new(
            batch.schema().fields().clone(),
            batch.columns().to_vec(),
            None,
        );
        let data = struct_array.to_data();
        let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&data).map_err(CudfError::Arrow)?;

        let schema_ptr = Box::into_raw(Box::new(ffi_schema)) as u64;
        let array_ptr = Box::into_raw(Box::new(ffi_array)) as u64;

        let result = cudf_cxx::interop::ffi::table_from_arrow_cdata(schema_ptr, array_ptr);
        if result.is_err() {
            // C++ threw before taking ownership — reclaim and drop to prevent leak.
            unsafe {
                drop(Box::from_raw(
                    schema_ptr as *mut arrow::ffi::FFI_ArrowSchema,
                ));
                drop(Box::from_raw(array_ptr as *mut arrow::ffi::FFI_ArrowArray));
            }
        }
        let raw = result.map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }
}

/// Legacy IPC-based conversion (requires `arrow-interop` feature).
#[cfg(feature = "arrow-interop")]
mod arrow_conv {
    use arrow::ipc;
    use arrow::record_batch::RecordBatch;

    use crate::error::{CudfError, Result};
    use crate::table::Table;

    impl Table {
        /// Convert this table to an `arrow::RecordBatch` via IPC.
        ///
        /// Consider using [`Table::to_arrow_batch`] instead for better
        /// performance (avoids IPC serialization overhead).
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

        /// Create a table from an `arrow::RecordBatch` via IPC.
        ///
        /// Consider using [`Table::from_arrow_batch`] instead for better
        /// performance (avoids IPC serialization overhead).
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

// ── DLPack ────────────────────────────────────────────────────

/// Opaque handle to a DLPack `DLManagedTensor`.
///
/// This is a GPU tensor representation used for zero-copy exchange with
/// frameworks like PyTorch and CuPy.
///
/// # Requirements
///
/// - All columns must have the same numeric data type.
/// - Null count must be zero for all columns.
pub struct DLPackTensor {
    ptr: u64,
}

impl DLPackTensor {
    /// Convert a table to a DLPack tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if columns have different types, non-numeric types,
    /// or non-zero null counts.
    pub fn from_table(table: &Table) -> Result<Self> {
        let ptr =
            cudf_cxx::interop::ffi::table_to_dlpack(&table.inner).map_err(CudfError::from_cxx)?;
        Ok(Self { ptr })
    }

    /// Import a table from this DLPack tensor, consuming the tensor.
    pub fn to_table(self) -> Result<Table> {
        // Wrap in ManuallyDrop immediately to prevent Drop from running
        // regardless of how this function exits (success, error, or panic).
        let this = std::mem::ManuallyDrop::new(self);
        match cudf_cxx::interop::ffi::table_from_dlpack(this.ptr) {
            Ok(raw) => {
                // Success: C++ consumed the tensor, don't run our Drop.
                Ok(Table { inner: raw })
            }
            Err(e) => {
                // C++ did not consume the tensor — free it manually.
                cudf_cxx::interop::ffi::free_dlpack(this.ptr);
                Err(CudfError::from_cxx(e))
            }
        }
    }

    /// Get the raw `DLManagedTensor*` pointer as `usize`.
    ///
    /// This is useful for passing to Python / C frameworks.
    /// The caller must NOT free the tensor; it is still owned by this handle.
    pub fn as_raw_ptr(&self) -> usize {
        self.ptr as usize
    }

    /// Create from a raw `DLManagedTensor*` pointer.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid `DLManagedTensor*` with a working deleter.
    pub unsafe fn from_raw_ptr(ptr: usize) -> Self {
        Self { ptr: ptr as u64 }
    }
}

impl Drop for DLPackTensor {
    fn drop(&mut self) {
        cudf_cxx::interop::ffi::free_dlpack(self.ptr);
    }
}

// ── contiguous_split / pack / unpack ──────────────────────────

/// A packed table: host-side metadata + a single contiguous GPU buffer.
///
/// This is the result of [`PackedTable::pack`] and is useful for:
/// - Efficient serialization (metadata + single memcpy)
/// - Reducing memory fragmentation
/// - IPC transfer between processes
pub struct PackedTable {
    inner: cxx::UniquePtr<cudf_cxx::interop::ffi::OwnedPackedColumns>,
}

impl PackedTable {
    /// Pack a table into a contiguous GPU buffer with host metadata.
    pub fn pack(table: &Table) -> Result<Self> {
        let raw = cudf_cxx::interop::ffi::pack_table(&table.inner).map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Get the host-side metadata bytes.
    ///
    /// This metadata describes column types, sizes, and offsets within
    /// the contiguous GPU buffer.
    pub fn metadata(&self) -> Result<Vec<u8>> {
        cudf_cxx::interop::ffi::packed_metadata(&self.inner).map_err(CudfError::from_cxx)
    }

    /// Get the size of the contiguous GPU data buffer in bytes.
    pub fn gpu_data_size(&self) -> Result<i64> {
        cudf_cxx::interop::ffi::packed_gpu_data_size(&self.inner).map_err(CudfError::from_cxx)
    }

    /// Unpack back into a table (deep copy from the contiguous buffer).
    pub fn unpack(&self) -> Result<Table> {
        let raw = cudf_cxx::interop::ffi::unpack_table(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}

/// Result of a contiguous_split operation.
///
/// Holds an opaque handle to the split partitions.  Individual partitions
/// can be extracted as [`PackedTable`]s.
///
/// **Important:** Each partition can only be extracted once via [`get`](SplitResult::get).
/// The underlying C++ implementation uses `std::move` to transfer ownership of the
/// packed data, so calling `get()` twice on the same index would yield moved-from
/// (invalid) data.  The `extracted` bitmap enforces this at runtime.
pub struct SplitResult {
    handle: u64,
    num_parts: usize,
    /// Tracks which partition indices have already been extracted.
    extracted: Vec<bool>,
}

impl SplitResult {
    /// Perform `contiguous_split` on a table at the given split indices.
    ///
    /// For N split points, produces N+1 partitions.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cudf::{Table, Column};
    /// use cudf::interop::SplitResult;
    ///
    /// let col = Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
    /// let table = Table::new(vec![col]).unwrap();
    /// let result = SplitResult::split(&table, &[2, 4]).unwrap();
    /// assert_eq!(result.num_parts(), 3);
    /// ```
    pub fn split(table: &Table, splits: &[i32]) -> Result<Self> {
        let info = cudf_cxx::interop::ffi::contiguous_split_table(&table.inner, splits)
            .map_err(CudfError::from_cxx)?;
        if info.len() < 2 {
            return Err(CudfError::InvalidArgument(
                "unexpected contiguous_split return".into(),
            ));
        }
        let num_parts = info[1] as usize;
        Ok(Self {
            handle: info[0],
            num_parts,
            extracted: vec![false; num_parts],
        })
    }

    /// Number of partitions in the result.
    pub fn num_parts(&self) -> usize {
        self.num_parts
    }

    /// Get a packed partition by index (moves data out of the split result).
    ///
    /// Each index can only be extracted **once**.  The underlying C++ code uses
    /// `std::move` on the partition data, so a second call with the same index
    /// would access moved-from (invalid) data.  This method tracks extractions
    /// and returns an error on duplicate access.
    pub fn get(&mut self, index: usize) -> Result<PackedTable> {
        if index >= self.num_parts {
            return Err(CudfError::IndexOutOfBounds {
                index,
                size: self.num_parts,
            });
        }
        if self.extracted[index] {
            return Err(CudfError::InvalidArgument(format!(
                "partition {index} has already been extracted from this SplitResult; \
                 each partition can only be retrieved once (the C++ side moves ownership)"
            )));
        }
        let raw = cudf_cxx::interop::ffi::contiguous_split_get(self.handle, checked_i32(index)?)
            .map_err(CudfError::from_cxx)?;
        self.extracted[index] = true;
        Ok(PackedTable { inner: raw })
    }
}

impl Drop for SplitResult {
    fn drop(&mut self) {
        cudf_cxx::interop::ffi::contiguous_split_free(self.handle);
    }
}
