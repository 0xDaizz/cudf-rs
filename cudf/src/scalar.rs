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

macro_rules! scalar_set_dispatch {
    ($type_id:expr, $value:expr, $inner:expr, $($variant:ident => $ty:ty, $set_fn:path);+ $(;)?) => {
        match $type_id {
            $(TypeId::$variant => {
                debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<$ty>());
                let v: $ty = unsafe { std::mem::transmute_copy(&$value) };
                $set_fn($inner.pin_mut(), v).map_err(CudfError::from_cxx)?;
            })+
            _ => {
                return Err(CudfError::InvalidArgument(format!(
                    "Scalar::new does not support {:?}",
                    $type_id
                )));
            }
        }
    };
}

macro_rules! scalar_get_dispatch {
    ($self_:expr, $($variant:ident => $ty:ty, $get_fn:path);+ $(;)?) => {
        match T::TYPE_ID {
            $(TypeId::$variant => {
                let v = $get_fn(&$self_.inner).map_err(CudfError::from_cxx)?;
                debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<$ty>());
                Ok(unsafe { std::mem::transmute_copy(&v) })
            })+
            _ => Err(CudfError::InvalidArgument(format!(
                "Scalar::value does not yet support {:?}",
                T::TYPE_ID
            ))),
        }
    };
}

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
        // so T IS the exact type we transmute to (same size, same repr).
        scalar_set_dispatch! {
            type_id, value, inner,
            Int8 => i8, cudf_cxx::scalar::ffi::scalar_set_i8;
            Int16 => i16, cudf_cxx::scalar::ffi::scalar_set_i16;
            Int32 => i32, cudf_cxx::scalar::ffi::scalar_set_i32;
            Int64 => i64, cudf_cxx::scalar::ffi::scalar_set_i64;
            Uint8 => u8, cudf_cxx::scalar::ffi::scalar_set_u8;
            Uint16 => u16, cudf_cxx::scalar::ffi::scalar_set_u16;
            Uint32 => u32, cudf_cxx::scalar::ffi::scalar_set_u32;
            Uint64 => u64, cudf_cxx::scalar::ffi::scalar_set_u64;
            Float32 => f32, cudf_cxx::scalar::ffi::scalar_set_f32;
            Float64 => f64, cudf_cxx::scalar::ffi::scalar_set_f64;
            Bool8 => bool, cudf_cxx::scalar::ffi::scalar_set_bool;
        }

        Ok(Self { inner })
    }

    /// Create a null scalar of the given data type.
    ///
    /// Note: temporal types (Timestamp*, Duration*) are not supported;
    /// use the C++ API directly for temporal null scalars.
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
        if !dtype.id().is_numeric() && dtype.id() != TypeId::Bool8 {
            return Err(CudfError::InvalidArgument(format!(
                "Scalar::null() only supports numeric and boolean types, got {:?}",
                dtype.id()
            )));
        }
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
        // CudfType trait, so transmute_copy is between identical types.
        scalar_get_dispatch! {
            self,
            Int8 => i8, cudf_cxx::scalar::ffi::scalar_get_i8;
            Int16 => i16, cudf_cxx::scalar::ffi::scalar_get_i16;
            Int32 => i32, cudf_cxx::scalar::ffi::scalar_get_i32;
            Int64 => i64, cudf_cxx::scalar::ffi::scalar_get_i64;
            Uint8 => u8, cudf_cxx::scalar::ffi::scalar_get_u8;
            Uint16 => u16, cudf_cxx::scalar::ffi::scalar_get_u16;
            Uint32 => u32, cudf_cxx::scalar::ffi::scalar_get_u32;
            Uint64 => u64, cudf_cxx::scalar::ffi::scalar_get_u64;
            Float32 => f32, cudf_cxx::scalar::ffi::scalar_get_f32;
            Float64 => f64, cudf_cxx::scalar::ffi::scalar_get_f64;
            Bool8 => bool, cudf_cxx::scalar::ffi::scalar_get_bool;
        }
    }

    /// Whether this scalar holds a valid (non-null) value.
    pub fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    /// Set the validity (null/not-null) of this scalar.
    pub fn set_valid(&mut self, valid: bool) -> Result<()> {
        cudf_cxx::scalar::ffi::scalar_set_valid(self.inner.pin_mut(), valid)
            .map_err(CudfError::from_cxx)
    }

    /// The data type of this scalar.
    pub fn data_type(&self) -> DataType {
        let raw = self.inner.type_id();
        let id = TypeId::from_raw(raw).unwrap_or_else(|| {
            panic!(
                "cudf: unrecognized type_id {} from FFI — possible libcudf version mismatch",
                raw
            )
        });
        DataType::new(id)
    }
}

// ── Convenience TryFrom impls ─────────────────────────────────

impl TryFrom<i32> for Scalar {
    type Error = CudfError;
    fn try_from(v: i32) -> Result<Self> {
        Self::new(v)
    }
}

impl TryFrom<i64> for Scalar {
    type Error = CudfError;
    fn try_from(v: i64) -> Result<Self> {
        Self::new(v)
    }
}

impl TryFrom<f32> for Scalar {
    type Error = CudfError;
    fn try_from(v: f32) -> Result<Self> {
        Self::new(v)
    }
}

impl TryFrom<f64> for Scalar {
    type Error = CudfError;
    fn try_from(v: f64) -> Result<Self> {
        Self::new(v)
    }
}

impl std::fmt::Debug for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::fmt::Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scalar({}, valid={})", self.data_type(), self.is_valid())
    }
}
