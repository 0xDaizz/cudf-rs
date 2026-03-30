//! GPU-resident column type.
//!
//! A [`Column`] owns GPU memory containing a single array of typed data,
//! optionally with a null bitmask. It is the fundamental building block
//! for GPU-accelerated DataFrame operations.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//!
//! // Create from a Rust slice (copies data to GPU)
//! let col = Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
//! assert_eq!(col.len(), 5);
//!
//! // Read back to host
//! let data: Vec<i32> = col.to_vec().unwrap();
//! assert_eq!(data, vec![1, 2, 3, 4, 5]);
//! ```

use std::fmt;

use cxx::UniquePtr;

use crate::error::{CudfError, Result};
use crate::types::{DataType, TypeId, checked_i32};

/// An owning, GPU-resident column of typed data.
///
/// `Column` wraps a `std::unique_ptr<cudf::column>` on the C++ side.
/// Dropping a `Column` frees the associated GPU memory.
///
/// # Thread Safety
///
/// `Column` implements [`Send`] (can be moved between threads) but not
/// [`Sync`] (cannot be shared between threads without synchronization).
/// Use `Arc<Mutex<Column>>` if shared access is needed.
pub struct Column {
    pub(crate) inner: UniquePtr<cudf_cxx::column::ffi::OwnedColumn>,
}

// SAFETY: GPU memory is process-global; a Column can be safely moved to another thread.
// CUDA operations must still be serialized per-stream, but Column ownership transfer is safe.
unsafe impl Send for Column {}

impl Column {
    // -- Accessors --

    /// Number of elements in this column.
    pub fn len(&self) -> usize {
        self.inner.size() as usize
    }

    /// Whether this column has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The data type of this column.
    pub fn data_type(&self) -> DataType {
        let id = TypeId::from_raw(self.inner.type_id()).unwrap_or(TypeId::Empty);
        let scale = self.inner.type_scale();
        if scale != 0 {
            DataType::decimal(id, scale).unwrap_or_else(|_| DataType::new(id))
        } else {
            DataType::new(id)
        }
    }

    /// Number of null elements. Returns 0 if the column is not nullable.
    pub fn null_count(&self) -> usize {
        self.inner.null_count() as usize
    }

    /// Whether this column can contain null values (has a validity bitmask).
    pub fn is_nullable(&self) -> bool {
        self.inner.is_nullable()
    }

    /// Whether this column actually contains any null values.
    pub fn has_nulls(&self) -> bool {
        self.inner.has_nulls()
    }

    /// Number of child columns (non-zero for nested types like LIST, STRUCT).
    pub fn num_children(&self) -> usize {
        self.inner.num_children() as usize
    }

    // -- Construction --

    /// Create an empty column with all nulls.
    ///
    /// Currently only numeric types are supported (the underlying C++ uses
    /// `cudf::make_numeric_column`). Non-numeric types will return an error.
    pub fn empty(dtype: DataType, size: usize) -> Result<Self> {
        if !dtype.id().is_numeric() {
            return Err(CudfError::InvalidArgument(format!(
                "Column::empty currently supports only numeric types, got {:?}",
                dtype.id()
            )));
        }
        let size_i32 = checked_i32(size)?;
        let raw = cudf_cxx::column::ffi::column_empty(dtype.id() as i32, size_i32)
            .map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Create a string column from a slice of string-like values, copying data to GPU.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::Column;
    ///
    /// let col = Column::from_strings(&["hello", "world", "!"]).unwrap();
    /// assert_eq!(col.len(), 3);
    /// ```
    pub fn from_strings(data: &[impl AsRef<str>]) -> Result<Self> {
        let strings: Vec<String> = data.iter().map(|s| s.as_ref().to_string()).collect();
        let raw =
            cudf_cxx::column::ffi::column_from_strings(&strings).map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Create a nullable i32 column from optional values.
    ///
    /// `None` values become null entries in the GPU column.
    pub fn from_optional_i32(data: &[Option<i32>]) -> Result<Self> {
        let values: Vec<i32> = data.iter().map(|o| o.unwrap_or(0)).collect();
        let validity: Vec<bool> = data.iter().map(|o| o.is_some()).collect();
        let raw = cudf_cxx::column::ffi::column_from_i32_nullable(&values, &validity)
            .map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Create a nullable i64 column from optional values.
    ///
    /// `None` values become null entries in the GPU column.
    pub fn from_optional_i64(data: &[Option<i64>]) -> Result<Self> {
        let values: Vec<i64> = data.iter().map(|o| o.unwrap_or(0)).collect();
        let validity: Vec<bool> = data.iter().map(|o| o.is_some()).collect();
        let raw = cudf_cxx::column::ffi::column_from_i64_nullable(&values, &validity)
            .map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Create a nullable f32 column from optional values.
    ///
    /// `None` values become null entries in the GPU column.
    pub fn from_optional_f32(data: &[Option<f32>]) -> Result<Self> {
        let values: Vec<f32> = data.iter().map(|o| o.unwrap_or(0.0)).collect();
        let validity: Vec<bool> = data.iter().map(|o| o.is_some()).collect();
        let raw = cudf_cxx::column::ffi::column_from_f32_nullable(&values, &validity)
            .map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Create a nullable f64 column from optional values.
    ///
    /// `None` values become null entries in the GPU column.
    pub fn from_optional_f64(data: &[Option<f64>]) -> Result<Self> {
        let values: Vec<f64> = data.iter().map(|o| o.unwrap_or(0.0)).collect();
        let validity: Vec<bool> = data.iter().map(|o| o.is_some()).collect();
        let raw = cudf_cxx::column::ffi::column_from_f64_nullable(&values, &validity)
            .map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Create a nullable i8 column from optional values.
    ///
    /// `None` values become null entries in the GPU column.
    pub fn from_optional_i8(data: &[Option<i8>]) -> Result<Self> {
        let values: Vec<i8> = data.iter().map(|o| o.unwrap_or(0)).collect();
        let validity: Vec<bool> = data.iter().map(|o| o.is_some()).collect();
        let raw = cudf_cxx::column::ffi::column_from_i8_nullable(&values, &validity)
            .map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Create a nullable i16 column from optional values.
    ///
    /// `None` values become null entries in the GPU column.
    pub fn from_optional_i16(data: &[Option<i16>]) -> Result<Self> {
        let values: Vec<i16> = data.iter().map(|o| o.unwrap_or(0)).collect();
        let validity: Vec<bool> = data.iter().map(|o| o.is_some()).collect();
        let raw = cudf_cxx::column::ffi::column_from_i16_nullable(&values, &validity)
            .map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Create a nullable u8 column from optional values.
    ///
    /// `None` values become null entries in the GPU column.
    pub fn from_optional_u8(data: &[Option<u8>]) -> Result<Self> {
        let values: Vec<u8> = data.iter().map(|o| o.unwrap_or(0)).collect();
        let validity: Vec<bool> = data.iter().map(|o| o.is_some()).collect();
        let raw = cudf_cxx::column::ffi::column_from_u8_nullable(&values, &validity)
            .map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Create a nullable u16 column from optional values.
    ///
    /// `None` values become null entries in the GPU column.
    pub fn from_optional_u16(data: &[Option<u16>]) -> Result<Self> {
        let values: Vec<u16> = data.iter().map(|o| o.unwrap_or(0)).collect();
        let validity: Vec<bool> = data.iter().map(|o| o.is_some()).collect();
        let raw = cudf_cxx::column::ffi::column_from_u16_nullable(&values, &validity)
            .map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Create a nullable u32 column from optional values.
    ///
    /// `None` values become null entries in the GPU column.
    pub fn from_optional_u32(data: &[Option<u32>]) -> Result<Self> {
        let values: Vec<u32> = data.iter().map(|o| o.unwrap_or(0)).collect();
        let validity: Vec<bool> = data.iter().map(|o| o.is_some()).collect();
        let raw = cudf_cxx::column::ffi::column_from_u32_nullable(&values, &validity)
            .map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Create a nullable u64 column from optional values.
    ///
    /// `None` values become null entries in the GPU column.
    pub fn from_optional_u64(data: &[Option<u64>]) -> Result<Self> {
        let values: Vec<u64> = data.iter().map(|o| o.unwrap_or(0)).collect();
        let validity: Vec<bool> = data.iter().map(|o| o.is_some()).collect();
        let raw = cudf_cxx::column::ffi::column_from_u64_nullable(&values, &validity)
            .map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Create a nullable string column from optional values.
    ///
    /// `None` values become null entries in the GPU column.
    /// For `None` entries, an empty string is stored as a placeholder;
    /// the null bitmask records which entries are actually null.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::Column;
    ///
    /// let col = Column::from_optional_strings(&[Some("hello"), None, Some("world")]).unwrap();
    /// assert_eq!(col.len(), 3);
    /// assert_eq!(col.null_count(), 1);
    /// ```
    pub fn from_optional_strings(data: &[Option<impl AsRef<str>>]) -> Result<Self> {
        let strings: Vec<String> = data
            .iter()
            .map(|o| match o {
                Some(s) => s.as_ref().to_string(),
                None => String::new(),
            })
            .collect();
        let validity: Vec<bool> = data.iter().map(|o| o.is_some()).collect();
        let raw = cudf_cxx::column::ffi::column_from_strings_nullable(&strings, &validity)
            .map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    /// Create a nullable bool column from optional values.
    ///
    /// `None` values become null entries in the GPU column.
    pub fn from_optional_bool(data: &[Option<bool>]) -> Result<Self> {
        let values: Vec<bool> = data.iter().map(|o| o.unwrap_or(false)).collect();
        let validity: Vec<bool> = data.iter().map(|o| o.is_some()).collect();
        let raw = cudf_cxx::column::ffi::column_from_bool_nullable(&values, &validity)
            .map_err(CudfError::from_cxx)?;
        Ok(Self { inner: raw })
    }

    // -- String Data Transfer --

    /// Extract all strings from a string column to host.
    ///
    /// Returns `Err` if the column is not a `STRING` type.
    /// For nullable columns, null entries are returned as empty strings.
    /// Use [`null_mask_to_host`](Self::null_mask_to_host) to distinguish nulls from
    /// actual empty strings.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::Column;
    ///
    /// let col = Column::from_strings(&["hello", "world"]).unwrap();
    /// let strings = col.to_strings().unwrap();
    /// assert_eq!(strings, vec!["hello".to_string(), "world".to_string()]);
    /// ```
    pub fn to_strings(&self) -> Result<Vec<String>> {
        cudf_cxx::column::ffi::column_to_strings(&self.inner).map_err(CudfError::from_cxx)
    }

    /// Extract all strings from a nullable string column as `Option<String>`.
    ///
    /// Valid entries are wrapped in `Some`, null entries become `None`.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::Column;
    ///
    /// let col = Column::from_optional_strings(&[Some("hello"), None, Some("world")]).unwrap();
    /// let strings = col.to_optional_strings().unwrap();
    /// assert_eq!(strings, vec![Some("hello".to_string()), None, Some("world".to_string())]);
    /// ```
    pub fn to_optional_strings(&self) -> Result<Vec<Option<String>>> {
        let strings = self.to_strings()?;

        if !self.has_nulls() {
            return Ok(strings.into_iter().map(Some).collect());
        }

        let mask = self.null_mask_to_host()?;
        Ok(strings
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                if mask[i / 8] & (1 << (i % 8)) != 0 {
                    Some(s)
                } else {
                    None
                }
            })
            .collect())
    }

    // -- Data Transfer --

    /// Copy the null bitmask to a host byte vector.
    /// Each bit indicates whether the corresponding element is valid (1) or null (0).
    ///
    /// The buffer is sized to match libcudf's `bitmask_allocation_size_bytes`,
    /// which pads to 64-byte alignment.
    pub fn null_mask_to_host(&self) -> Result<Vec<u8>> {
        // libcudf's bitmask_allocation_size_bytes pads to 64-byte boundaries.
        // We must match that to avoid "Output buffer too small" from the C++ side.
        let num_bits_bytes = self.len().div_ceil(8);
        let num_bytes = (num_bits_bytes + 63) & !63; // pad to 64-byte alignment
        let num_bytes = num_bytes.max(64); // minimum 64 bytes (matches cudf policy)
        let mut buf = vec![0u8; num_bytes];
        cudf_cxx::column::ffi::column_null_mask(&self.inner, &mut buf)
            .map_err(CudfError::from_cxx)?;
        // Truncate to only the meaningful bytes
        buf.truncate(self.len().div_ceil(8));
        Ok(buf)
    }
}

// -- Type-specific construction and transfer --

mod private {
    pub trait Sealed {}
}

impl private::Sealed for i8 {}
impl private::Sealed for i16 {}
impl private::Sealed for i32 {}
impl private::Sealed for i64 {}
impl private::Sealed for u8 {}
impl private::Sealed for u16 {}
impl private::Sealed for u32 {}
impl private::Sealed for u64 {}
impl private::Sealed for f32 {}
impl private::Sealed for f64 {}
impl private::Sealed for bool {}

/// Trait for types that can be stored in a GPU column.
///
/// This is implemented for all primitive numeric types supported by libcudf.
/// It enables generic `Column::from_slice` and `Column::to_vec` operations.
///
/// This trait is sealed and cannot be implemented outside of the `cudf` crate.
pub trait CudfType: Copy + Send + 'static + private::Sealed {
    /// The corresponding libcudf type ID.
    const TYPE_ID: TypeId;
}

// Register all supported types
impl CudfType for i8 {
    const TYPE_ID: TypeId = TypeId::Int8;
}
impl CudfType for i16 {
    const TYPE_ID: TypeId = TypeId::Int16;
}
impl CudfType for i32 {
    const TYPE_ID: TypeId = TypeId::Int32;
}
impl CudfType for i64 {
    const TYPE_ID: TypeId = TypeId::Int64;
}
impl CudfType for u8 {
    const TYPE_ID: TypeId = TypeId::Uint8;
}
impl CudfType for u16 {
    const TYPE_ID: TypeId = TypeId::Uint16;
}
impl CudfType for u32 {
    const TYPE_ID: TypeId = TypeId::Uint32;
}
impl CudfType for u64 {
    const TYPE_ID: TypeId = TypeId::Uint64;
}
impl CudfType for f32 {
    const TYPE_ID: TypeId = TypeId::Float32;
}
impl CudfType for f64 {
    const TYPE_ID: TypeId = TypeId::Float64;
}
impl CudfType for bool {
    const TYPE_ID: TypeId = TypeId::Bool8;
}

impl Column {
    /// Create a column from a host slice, copying data to GPU.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::Column;
    ///
    /// let ints = Column::from_slice(&[1i32, 2, 3]).unwrap();
    /// let floats = Column::from_slice(&[1.0f64, 2.0, 3.0]).unwrap();
    /// ```
    pub fn from_slice<T: CudfType>(data: &[T]) -> Result<Self> {
        let inner = match T::TYPE_ID {
            TypeId::Int8 => {
                // SAFETY: T is i8 when TYPE_ID is Int8; same size and repr.
                let data =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i8, data.len()) };
                cudf_cxx::column::ffi::column_from_i8(data)
            }
            TypeId::Int16 => {
                let data =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i16, data.len()) };
                cudf_cxx::column::ffi::column_from_i16(data)
            }
            TypeId::Int32 => {
                let data =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, data.len()) };
                cudf_cxx::column::ffi::column_from_i32(data)
            }
            TypeId::Int64 => {
                let data =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i64, data.len()) };
                cudf_cxx::column::ffi::column_from_i64(data)
            }
            TypeId::Uint8 => {
                let data =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
                cudf_cxx::column::ffi::column_from_u8(data)
            }
            TypeId::Uint16 => {
                let data =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, data.len()) };
                cudf_cxx::column::ffi::column_from_u16(data)
            }
            TypeId::Uint32 => {
                let data =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len()) };
                cudf_cxx::column::ffi::column_from_u32(data)
            }
            TypeId::Uint64 => {
                let data =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u64, data.len()) };
                cudf_cxx::column::ffi::column_from_u64(data)
            }
            TypeId::Float32 => {
                let data =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
                cudf_cxx::column::ffi::column_from_f32(data)
            }
            TypeId::Float64 => {
                let data =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, data.len()) };
                cudf_cxx::column::ffi::column_from_f64(data)
            }
            TypeId::Bool8 => {
                let data =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const bool, data.len()) };
                cudf_cxx::column::ffi::column_from_bool(data)
            }
            _ => {
                return Err(CudfError::InvalidArgument(format!(
                    "from_slice does not support {:?}",
                    T::TYPE_ID
                )));
            }
        }
        .map_err(CudfError::from_cxx)?;

        Ok(Self { inner })
    }

    /// Copy column data back to host as a Vec.
    ///
    /// # Type Safety
    ///
    /// The type parameter `T` must match the column's actual data type.
    /// Returns `Err(CudfError::TypeMismatch)` if they don't match.
    ///
    /// # Nullability
    ///
    /// Returns `Err(CudfError::InvalidArgument)` if the column contains nulls,
    /// because null values would be silently replaced by whatever GPU memory
    /// happens to contain (indistinguishable from real values).
    /// Use [`to_optional_vec`](Self::to_optional_vec) instead for nullable columns.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::Column;
    ///
    /// let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    /// let data: Vec<i32> = col.to_vec().unwrap();
    /// assert_eq!(data, vec![1, 2, 3]);
    /// ```
    pub fn to_vec<T: CudfType>(&self) -> Result<Vec<T>> {
        if self.has_nulls() {
            return Err(CudfError::InvalidArgument(
                "to_vec() cannot be used on columns with null values (null values would be \
                 indistinguishable from real data). Use to_optional_vec() instead, or use \
                 null_mask_to_host() to get the null mask separately."
                    .into(),
            ));
        }
        self.to_vec_raw()
    }

    /// Copy column data back to host as a `Vec<Option<T>>`, preserving null information.
    ///
    /// Valid elements are wrapped in `Some`, null elements become `None`.
    /// If the column has no nulls, all elements will be `Some`.
    ///
    /// # Type Safety
    ///
    /// The type parameter `T` must match the column's actual data type.
    /// Returns `Err(CudfError::TypeMismatch)` if they don't match.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::Column;
    ///
    /// let col = Column::from_optional_i32(&[Some(1), None, Some(3)]).unwrap();
    /// let data: Vec<Option<i32>> = col.to_optional_vec().unwrap();
    /// assert_eq!(data, vec![Some(1), None, Some(3)]);
    /// ```
    pub fn to_optional_vec<T: CudfType>(&self) -> Result<Vec<Option<T>>> {
        let values = self.to_vec_raw::<T>()?;

        if !self.has_nulls() {
            return Ok(values.into_iter().map(Some).collect());
        }

        let mask = self.null_mask_to_host()?;
        Ok(values
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                if mask[i / 8] & (1 << (i % 8)) != 0 {
                    Some(v)
                } else {
                    None
                }
            })
            .collect())
    }

    /// Internal: transfer column data to host without null checking.
    fn to_vec_raw<T: CudfType>(&self) -> Result<Vec<T>> {
        // Type check
        let actual = self.data_type().id();
        if actual != T::TYPE_ID {
            return Err(CudfError::TypeMismatch {
                expected: format!("{:?}", T::TYPE_ID),
                actual: format!("{:?}", actual),
            });
        }

        let len = self.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        // Allocate output buffer. We write into spare capacity via raw pointer,
        // only calling set_len AFTER the C++ side successfully fills the data.
        let mut result: Vec<T> = Vec::with_capacity(len);
        let ptr = result.as_mut_ptr();

        // Dispatch to the appropriate cudf-cxx transfer function.
        match T::TYPE_ID {
            TypeId::Int8 => {
                let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i8, len) };
                cudf_cxx::column::ffi::column_to_i8(&self.inner, out)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Int16 => {
                let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i16, len) };
                cudf_cxx::column::ffi::column_to_i16(&self.inner, out)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Int32 => {
                let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i32, len) };
                cudf_cxx::column::ffi::column_to_i32(&self.inner, out)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Int64 => {
                let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i64, len) };
                cudf_cxx::column::ffi::column_to_i64(&self.inner, out)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Uint8 => {
                let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, len) };
                cudf_cxx::column::ffi::column_to_u8(&self.inner, out)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Uint16 => {
                let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u16, len) };
                cudf_cxx::column::ffi::column_to_u16(&self.inner, out)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Uint32 => {
                let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, len) };
                cudf_cxx::column::ffi::column_to_u32(&self.inner, out)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Uint64 => {
                let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u64, len) };
                cudf_cxx::column::ffi::column_to_u64(&self.inner, out)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Float32 => {
                let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, len) };
                cudf_cxx::column::ffi::column_to_f32(&self.inner, out)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Float64 => {
                let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f64, len) };
                cudf_cxx::column::ffi::column_to_f64(&self.inner, out)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Bool8 => {
                // Read into a temporary u8 buffer, then convert to bool safely.
                // Direct write to Vec<bool> memory is UB if GPU data contains values != 0/1.
                let mut u8_buf = vec![0u8; len];
                cudf_cxx::column::ffi::column_to_u8(&self.inner, &mut u8_buf)
                    .map_err(CudfError::from_cxx)?;
                let bools: Vec<bool> = u8_buf.into_iter().map(|v| v != 0).collect();
                // SAFETY: T is guaranteed to be bool by the sealed CudfType trait
                // (Bool8 maps only to bool). Vec<bool> and Vec<T=bool> have
                // identical layout. We use manual deconstruction to avoid
                // relying on transmute's layout guarantees for Vec.
                let mut bools = std::mem::ManuallyDrop::new(bools);
                let result = unsafe {
                    Vec::from_raw_parts(
                        bools.as_mut_ptr() as *mut T,
                        bools.len(),
                        bools.capacity(),
                    )
                };
                return Ok(result);
            }
            _ => {
                return Err(CudfError::InvalidArgument(format!(
                    "to_vec does not yet support {:?}",
                    T::TYPE_ID
                )));
            }
        }

        // SAFETY: The C++ side has successfully filled exactly `len` elements
        // via cudaMemcpy. The type size is guaranteed correct by the type check above.
        unsafe {
            result.set_len(len);
        }
        Ok(result)
    }
}

impl fmt::Debug for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Column({}, len={}, nulls={})",
            self.data_type(),
            self.len(),
            self.null_count()
        )
    }
}
