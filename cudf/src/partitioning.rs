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
//! let result = table.hash_partition(&[0], 3).unwrap();
//! let _table = result.table;
//! let _offsets = result.offsets;
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;
use crate::types::checked_i32;

/// Result of a partition operation.
pub struct PartitionResult {
    /// The reordered table with rows grouped by partition.
    pub table: Table,
    /// Offsets to the start of each partition. Partition `i` spans
    /// rows `offsets[i]..offsets[i+1]`.
    pub offsets: Vec<i32>,
}

impl Table {
    /// Partition this table by hashing the specified columns.
    ///
    /// Returns a reordered table where rows are grouped by partition.
    ///
    /// # Arguments
    ///
    /// * `columns_to_hash` - Column indices to hash for partitioning.
    /// * `num_partitions` - Number of partitions to create.
    pub fn hash_partition(
        &self,
        columns_to_hash: &[usize],
        num_partitions: usize,
    ) -> Result<PartitionResult> {
        let cols: Vec<i32> = columns_to_hash
            .iter()
            .map(|&c| checked_i32(c))
            .collect::<Result<Vec<i32>>>()?;
        let result = cudf_cxx::partitioning::ffi::hash_partition(
            &self.inner,
            &cols,
            checked_i32(num_partitions)?,
        )
        .map_err(CudfError::from_cxx)?;

        let offsets = cudf_cxx::partitioning::ffi::partition_result_offsets(&result);
        let table_raw = cudf_cxx::partitioning::ffi::partition_result_table(result)
            .map_err(CudfError::from_cxx)?;

        Ok(PartitionResult {
            table: Table { inner: table_raw },
            offsets,
        })
    }

    /// Partition this table using round-robin assignment.
    ///
    /// Rows are distributed evenly across partitions in order.
    pub fn round_robin_partition(&self, num_partitions: usize) -> Result<PartitionResult> {
        let result = cudf_cxx::partitioning::ffi::round_robin_partition(
            &self.inner,
            checked_i32(num_partitions)?,
        )
        .map_err(CudfError::from_cxx)?;

        let offsets = cudf_cxx::partitioning::ffi::partition_result_offsets(&result);
        let table_raw = cudf_cxx::partitioning::ffi::partition_result_table(result)
            .map_err(CudfError::from_cxx)?;

        Ok(PartitionResult {
            table: Table { inner: table_raw },
            offsets,
        })
    }

    /// Partition this table using a partition map column.
    ///
    /// Each element in `partition_map` specifies which partition (0..num_partitions)
    /// the corresponding row belongs to. Returns a [`PartitionResult`] with
    /// the reordered table and partition offsets.
    ///
    /// # Arguments
    ///
    /// * `partition_map` - Integer column mapping each row to a partition.
    /// * `num_partitions` - Number of partitions to create.
    pub fn partition(
        &self,
        partition_map: &Column,
        num_partitions: usize,
    ) -> Result<PartitionResult> {
        let result = cudf_cxx::partitioning::ffi::partition(
            &self.inner,
            &partition_map.inner,
            checked_i32(num_partitions)?,
        )
        .map_err(CudfError::from_cxx)?;

        let offsets = cudf_cxx::partitioning::ffi::partition_result_offsets(&result);
        let table_raw = cudf_cxx::partitioning::ffi::partition_result_table(result)
            .map_err(CudfError::from_cxx)?;

        Ok(PartitionResult {
            table: Table { inner: table_raw },
            offsets,
        })
    }
}
