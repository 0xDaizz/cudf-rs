//! GPU-resident scalar type.
//!
//! A [`Scalar`] owns GPU memory containing a single typed value,
//! optionally marked as null. Scalars are used as arguments to
//! binary operations (column-scalar) and other libcudf functions.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Scalar;
//!
//! let s = Scalar::new(42i32).unwrap();
//! assert!(s.is_valid());
//! assert_eq!(s.value::<i32>().unwrap(), 42);
//! ```

use cxx::UniquePtr;

use crate::column::CudfType;
use crate::error::{CudfError, Result};
use crate::types::{DataType, TypeId};

/// An owning, GPU-resident scalar value.
///
/// `Scalar` wraps a `std::unique_ptr<cudf::scalar>` on the C++ side.
/// Dropping a `Scalar` frees the associated GPU memory.
///
/// # Thread Safety
///
/// `Scalar` implements [`Send`] (can be moved between threads) but not
/// [`Sync`] (cannot be shared between threads without synchronization).
pub struct Scalar {
    pub(crate) inner: UniquePtr<cudf_cxx::scalar::ffi::OwnedScalar>,
}

// SAFETY: GPU memory is process-global; a Scalar can be safely moved to another thread.
unsafe impl Send for Scalar {}

impl Scalar {
    /// Create a scalar with the given value.
    ///
    /// The scalar is marked as valid (non-null).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::Scalar;
    ///
    /// let s = Scalar::new(3.14f64).unwrap();
    /// assert!(s.is_valid());
    /// ```
    pub fn new<T: CudfType>(value: T) -> Result<Self> {
        let type_id = T::TYPE_ID;
        let mut inner = cudf_cxx::scalar::ffi::make_numeric_scalar(type_id as i32)
            .map_err(CudfError::from_cxx)?;

        // Set the value via the appropriate typed setter.
        // SAFETY for each arm: CudfType guarantees T matches TYPE_ID,
        // so T IS the exact type we cast to (same size, same repr).
        match type_id {
            TypeId::Int8 => {
                let v = unsafe { *(&value as *const T as *const i8) };
                cudf_cxx::scalar::ffi::scalar_set_i8(inner.pin_mut(), v)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Int16 => {
                let v = unsafe { *(&value as *const T as *const i16) };
                cudf_cxx::scalar::ffi::scalar_set_i16(inner.pin_mut(), v)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Int32 => {
                let v = unsafe { *(&value as *const T as *const i32) };
                cudf_cxx::scalar::ffi::scalar_set_i32(inner.pin_mut(), v)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Int64 => {
                let v = unsafe { *(&value as *const T as *const i64) };
                cudf_cxx::scalar::ffi::scalar_set_i64(inner.pin_mut(), v)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Uint8 => {
                let v = unsafe { *(&value as *const T as *const u8) };
                cudf_cxx::scalar::ffi::scalar_set_u8(inner.pin_mut(), v)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Uint16 => {
                let v = unsafe { *(&value as *const T as *const u16) };
                cudf_cxx::scalar::ffi::scalar_set_u16(inner.pin_mut(), v)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Uint32 => {
                let v = unsafe { *(&value as *const T as *const u32) };
                cudf_cxx::scalar::ffi::scalar_set_u32(inner.pin_mut(), v)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Uint64 => {
                let v = unsafe { *(&value as *const T as *const u64) };
                cudf_cxx::scalar::ffi::scalar_set_u64(inner.pin_mut(), v)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Float32 => {
                let v = unsafe { *(&value as *const T as *const f32) };
                cudf_cxx::scalar::ffi::scalar_set_f32(inner.pin_mut(), v)
                    .map_err(CudfError::from_cxx)?;
            }
            TypeId::Float64 => {
                let v = unsafe { *(&value as *const T as *const f64) };
                cudf_cxx::scalar::ffi::scalar_set_f64(inner.pin_mut(), v)
                    .map_err(CudfError::from_cxx)?;
            }
            _ => {
                return Err(CudfError::InvalidArgument(format!(
                    "Scalar::new does not support {:?}",
                    type_id
                )));
            }
        }

        Ok(Self { inner })
    }

    /// Create a null scalar of the given data type.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::{Scalar, DataType, TypeId};
    ///
    /// let s = Scalar::null(DataType::new(TypeId::Int32)).unwrap();
    /// assert!(!s.is_valid());
    /// ```
    pub fn null(dtype: DataType) -> Result<Self> {
        let inner = cudf_cxx::scalar::ffi::make_numeric_scalar(dtype.id() as i32)
            .map_err(CudfError::from_cxx)?;
        // Scalar is already invalid (null) by default from make_numeric_scalar
        Ok(Self { inner })
    }

    /// Get the value of this scalar.
    ///
    /// # Type Safety
    ///
    /// The type parameter `T` must match the scalar's actual data type.
    /// Returns `Err(CudfError::TypeMismatch)` if they don't match.
    pub fn value<T: CudfType>(&self) -> Result<T> {
        let actual = self.data_type().id();
        if actual != T::TYPE_ID {
            return Err(CudfError::TypeMismatch {
                expected: format!("{:?}", T::TYPE_ID),
                actual: format!("{:?}", actual),
            });
        }

        if !self.is_valid() {
            return Err(CudfError::InvalidArgument(
                "cannot get value of null scalar".to_string(),
            ));
        }

        // SAFETY for each arm: T is guaranteed to match TYPE_ID by the
        // CudfType trait, so the pointer cast is between identical types.
        match T::TYPE_ID {
            TypeId::Int8 => {
                let v = cudf_cxx::scalar::ffi::scalar_get_i8(&self.inner)
                    .map_err(CudfError::from_cxx)?;
                Ok(unsafe { *(&v as *const i8 as *const T) })
            }
            TypeId::Int16 => {
                let v = cudf_cxx::scalar::ffi::scalar_get_i16(&self.inner)
                    .map_err(CudfError::from_cxx)?;
                Ok(unsafe { *(&v as *const i16 as *const T) })
            }
            TypeId::Int32 => {
                let v = cudf_cxx::scalar::ffi::scalar_get_i32(&self.inner)
                    .map_err(CudfError::from_cxx)?;
                Ok(unsafe { *(&v as *const i32 as *const T) })
            }
            TypeId::Int64 => {
                let v = cudf_cxx::scalar::ffi::scalar_get_i64(&self.inner)
                    .map_err(CudfError::from_cxx)?;
                Ok(unsafe { *(&v as *const i64 as *const T) })
            }
            TypeId::Uint8 => {
                let v = cudf_cxx::scalar::ffi::scalar_get_u8(&self.inner)
                    .map_err(CudfError::from_cxx)?;
                Ok(unsafe { *(&v as *const u8 as *const T) })
            }
            TypeId::Uint16 => {
                let v = cudf_cxx::scalar::ffi::scalar_get_u16(&self.inner)
                    .map_err(CudfError::from_cxx)?;
                Ok(unsafe { *(&v as *const u16 as *const T) })
            }
            TypeId::Uint32 => {
                let v = cudf_cxx::scalar::ffi::scalar_get_u32(&self.inner)
                    .map_err(CudfError::from_cxx)?;
                Ok(unsafe { *(&v as *const u32 as *const T) })
            }
            TypeId::Uint64 => {
                let v = cudf_cxx::scalar::ffi::scalar_get_u64(&self.inner)
                    .map_err(CudfError::from_cxx)?;
                Ok(unsafe { *(&v as *const u64 as *const T) })
            }
            TypeId::Float32 => {
                let v = cudf_cxx::scalar::ffi::scalar_get_f32(&self.inner)
                    .map_err(CudfError::from_cxx)?;
                Ok(unsafe { *(&v as *const f32 as *const T) })
            }
            TypeId::Float64 => {
                let v = cudf_cxx::scalar::ffi::scalar_get_f64(&self.inner)
                    .map_err(CudfError::from_cxx)?;
                Ok(unsafe { *(&v as *const f64 as *const T) })
            }
            _ => Err(CudfError::InvalidArgument(format!(
                "Scalar::value does not yet support {:?}",
                T::TYPE_ID
            ))),
        }
    }

    /// Whether this scalar holds a valid (non-null) value.
    pub fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    /// The data type of this scalar.
    pub fn data_type(&self) -> DataType {
        let id = TypeId::from_raw(self.inner.type_id()).unwrap_or(TypeId::Empty);
        DataType::new(id)
    }
}

// ── Convenience TryFrom impls ─────────────────────────────────

impl TryFrom<i32> for Scalar {
    type Error = CudfError;
    fn try_from(v: i32) -> Result<Self> { Self::new(v) }
}

impl TryFrom<i64> for Scalar {
    type Error = CudfError;
    fn try_from(v: i64) -> Result<Self> { Self::new(v) }
}

impl TryFrom<f32> for Scalar {
    type Error = CudfError;
    fn try_from(v: f32) -> Result<Self> { Self::new(v) }
}

impl TryFrom<f64> for Scalar {
    type Error = CudfError;
    fn try_from(v: f64) -> Result<Self> { Self::new(v) }
}

impl std::fmt::Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Scalar({}, valid={})",
            self.data_type(),
            self.is_valid()
        )
    }
}
