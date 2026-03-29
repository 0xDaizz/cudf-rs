//! GPU-accelerated partitioning operations.
//!
//! Provides hash and round-robin partitioning for [`Table`]s.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//!
//! let col = Column::from_slice(&[1i32, 2, 3, 4, 5, 6]).unwrap();
//! let table = Table::new(vec![col]).unwrap();
//!
//! let partitioned = table.hash_partition(&[0], 3).unwrap();
//! ```

use crate::error::{CudfError, Result};
use crate::table::Table;

impl Table {
    /// Partition this table by hashing the specified columns.
    ///
    /// Returns a reordered table where rows are grouped by partition.
    ///
    /// # Arguments
    ///
    /// * `columns_to_hash` - Column indices to hash for partitioning.
    /// * `num_partitions` - Number of partitions to create.
    pub fn hash_partition(&self, columns_to_hash: &[i32], num_partitions: usize) -> Result<Table> {
        let raw = cudf_cxx::partitioning::ffi::hash_partition(
            &self.inner,
            columns_to_hash,
            num_partitions as i32,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }

    /// Partition this table using round-robin assignment.
    ///
    /// Rows are distributed evenly across partitions in order.
    pub fn round_robin_partition(&self, num_partitions: usize) -> Result<Table> {
        let raw =
            cudf_cxx::partitioning::ffi::round_robin_partition(&self.inner, num_partitions as i32)
                .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}
