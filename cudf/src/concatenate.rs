//! Concatenation operations for columns and tables.
//!
//! Provides GPU-accelerated vertical concatenation (stacking) of
//! columns and tables.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//! use cudf::concatenate;
//!
//! let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
//! let b = Column::from_slice(&[4i32, 5, 6]).unwrap();
//! let combined = concatenate::concatenate_columns(&[&a, &b]).unwrap();
//! assert_eq!(combined.len(), 6);
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;

/// Concatenate multiple columns vertically into a single column.
///
/// All columns must have the same data type. The result is a new column
/// containing all elements from the input columns in order.
///
/// # Errors
///
/// Returns an error if:
/// - The input slice is empty
/// - The columns have mismatched types
/// - A GPU error occurs
pub fn concatenate_columns(columns: &[&Column]) -> Result<Column> {
    if columns.is_empty() {
        return Err(CudfError::InvalidArgument(
            "Cannot concatenate zero columns".to_string(),
        ));
    }

    let mut builder = cudf_cxx::concatenate::ffi::new_column_concat_builder();
    for col in columns {
        builder.as_mut().unwrap().add(&col.inner);
    }
    let raw = builder.build().map_err(CudfError::from_cxx)?;
    Ok(Column { inner: raw })
}

/// Concatenate multiple tables vertically into a single table.
///
/// All tables must have the same number of columns with matching types.
/// The result is a new table with rows from all input tables stacked.
///
/// # Errors
///
/// Returns an error if:
/// - The input slice is empty
/// - The tables have mismatched schemas
/// - A GPU error occurs
pub fn concatenate_tables(tables: &[&Table]) -> Result<Table> {
    if tables.is_empty() {
        return Err(CudfError::InvalidArgument(
            "Cannot concatenate zero tables".to_string(),
        ));
    }

    let mut builder = cudf_cxx::concatenate::ffi::new_table_concat_builder();
    for tbl in tables {
        builder.as_mut().unwrap().add(&tbl.inner);
    }
    let raw = builder.build().map_err(CudfError::from_cxx)?;
    Ok(Table { inner: raw })
}
