//! Timezone transition table support.
//!
//! Used for converting ORC timestamps to UTC.

use crate::error::{CudfError, Result};
use crate::table::Table;

/// Create a timezone transition table for the given timezone name.
///
/// Uses the system's TZif files to build a transition table for
/// converting ORC timestamps to UTC.
///
/// # Arguments
///
/// * `timezone_name` - Standard timezone name (e.g., `"America/Los_Angeles"`)
pub fn make_timezone_transition_table(timezone_name: &str) -> Result<Table> {
    let result = cudf_cxx::timezone::ffi::make_timezone_transition_table(timezone_name)
        .map_err(CudfError::from_cxx)?;
    Ok(Table { inner: result })
}
