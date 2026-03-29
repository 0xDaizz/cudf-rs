//! Join operations on GPU tables.
//!
//! Joins match rows between two tables based on key equality.
//! Results are returned as gather maps (index columns) that can be
//! applied to the original tables via `Table::gather()`.
//! `cross_join` is the exception: it returns the full cartesian-product
//! table directly rather than gather maps.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//! use cudf::join::JoinResult;
//!
//! let left_keys = Table::new(vec![
//!     Column::from_slice(&[1i32, 2, 3, 4]).unwrap(),
//! ]).unwrap();
//! let right_keys = Table::new(vec![
//!     Column::from_slice(&[2i32, 3, 5]).unwrap(),
//! ]).unwrap();
//!
//! let result: JoinResult = left_keys.inner_join(&right_keys).unwrap();
//! // result.left_indices and result.right_indices are gather maps
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;

/// Result of a join operation: left and right gather maps.
///
/// Use these index columns with `Table::gather()` (or equivalent) to
/// construct the actual joined table from the original left and right tables.
pub struct JoinResult {
    /// Index column for gathering from the left table.
    pub left_indices: Column,
    /// Index column for gathering from the right table.
    pub right_indices: Column,
}

/// Result of a semi/anti join: a single gather map for the left table.
///
/// Use the index column with `Table::gather()` to construct the filtered
/// left table.
pub struct SemiJoinResult {
    /// Index column for gathering matching (or non-matching) rows from the left table.
    pub left_indices: Column,
}

impl Table {
    /// Perform an inner join on key columns.
    ///
    /// Returns gather maps (index pairs) for constructing the joined table.
    /// Only rows with matching keys in both tables are included.
    ///
    /// # Errors
    ///
    /// Returns an error if column types don't match or a GPU error occurs.
    pub fn inner_join(&self, right_keys: &Table) -> Result<JoinResult> {
        let maps = cudf_cxx::join::ffi::inner_join(&self.inner, &right_keys.inner)
            .map_err(CudfError::from_cxx)?;
        extract_join_result(maps)
    }

    /// Perform a left join on key columns.
    ///
    /// All rows from the left table are preserved. Right-side indices will
    /// contain nulls for rows without a match.
    ///
    /// # Errors
    ///
    /// Returns an error if column types don't match or a GPU error occurs.
    pub fn left_join(&self, right_keys: &Table) -> Result<JoinResult> {
        let maps = cudf_cxx::join::ffi::left_join(&self.inner, &right_keys.inner)
            .map_err(CudfError::from_cxx)?;
        extract_join_result(maps)
    }

    /// Perform a full outer join on key columns.
    ///
    /// All rows from both tables are preserved. Indices will contain nulls
    /// where no match exists on either side.
    ///
    /// # Errors
    ///
    /// Returns an error if column types don't match or a GPU error occurs.
    pub fn full_join(&self, right_keys: &Table) -> Result<JoinResult> {
        let maps = cudf_cxx::join::ffi::full_join(&self.inner, &right_keys.inner)
            .map_err(CudfError::from_cxx)?;
        extract_join_result(maps)
    }

    /// Perform a left semi join on key columns.
    ///
    /// Returns a gather map of left-table row indices that have at least
    /// one matching row in the right table.
    ///
    /// # Errors
    ///
    /// Returns an error if column types don't match or a GPU error occurs.
    pub fn left_semi_join(&self, right_keys: &Table) -> Result<SemiJoinResult> {
        let maps = cudf_cxx::join::ffi::left_semi_join(&self.inner, &right_keys.inner)
            .map_err(CudfError::from_cxx)?;
        extract_semi_join_result(maps)
    }

    /// Perform a left anti join on key columns.
    ///
    /// Returns a gather map of left-table row indices that have NO
    /// matching row in the right table.
    ///
    /// # Errors
    ///
    /// Returns an error if column types don't match or a GPU error occurs.
    pub fn left_anti_join(&self, right_keys: &Table) -> Result<SemiJoinResult> {
        let maps = cudf_cxx::join::ffi::left_anti_join(&self.inner, &right_keys.inner)
            .map_err(CudfError::from_cxx)?;
        extract_semi_join_result(maps)
    }

    /// Cross join (cartesian product) of two tables.
    ///
    /// Returns a table containing every combination of rows from the left
    /// and right tables. The result has `left.num_rows() * right.num_rows()`
    /// rows and `left.num_columns() + right.num_columns()` columns.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn cross_join(&self, right: &Table) -> Result<Table> {
        let raw = cudf_cxx::join::ffi::cross_join(&self.inner, &right.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}

/// Extract the single index column from a 1-column gather map table (semi/anti join).
fn extract_semi_join_result(
    mut maps: cxx::UniquePtr<cudf_cxx::table::ffi::OwnedTable>,
) -> Result<SemiJoinResult> {
    let left = cudf_cxx::table::ffi::table_release_column(maps.pin_mut(), 0)
        .map_err(CudfError::from_cxx)?;
    Ok(SemiJoinResult {
        left_indices: Column { inner: left },
    })
}

/// Extract left and right index columns from a 2-column gather map table.
///
/// Uses `table_release_column` (zero-copy) instead of `table_get_column`
/// (which copies GPU memory). Columns are released in reverse order
/// (index 1 then 0) because each release shifts subsequent indices.
fn extract_join_result(
    mut maps: cxx::UniquePtr<cudf_cxx::table::ffi::OwnedTable>,
) -> Result<JoinResult> {
    let right = cudf_cxx::table::ffi::table_release_column(maps.pin_mut(), 1)
        .map_err(CudfError::from_cxx)?;
    let left = cudf_cxx::table::ffi::table_release_column(maps.pin_mut(), 0)
        .map_err(CudfError::from_cxx)?;
    Ok(JoinResult {
        left_indices: Column { inner: left },
        right_indices: Column { inner: right },
    })
}
