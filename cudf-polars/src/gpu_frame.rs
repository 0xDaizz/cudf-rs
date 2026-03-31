//! GpuDataFrame: a named collection of GPU-resident columns.
//!
//! Wraps a `cudf::Table` with column names, providing named access
//! and conversion back to Polars `DataFrame`.

use cudf::aggregation::AggregationKind;
use cudf::groupby::GroupBy;
use cudf::sorting::{NullOrder, SortOrder};
use cudf::stream_compaction::{DuplicateKeepOption, NullEquality};
use cudf::{Column as GpuColumn, Table as GpuTable};
use polars_core::prelude::*;
use polars_error::{PolarsResult, polars_err};

use crate::convert;
use crate::error::gpu_result;

/// A GPU-resident data frame with named columns.
pub struct GpuDataFrame {
    table: GpuTable,
    names: Vec<String>,
}

impl GpuDataFrame {
    /// Create from a cudf Table and corresponding column names.
    pub fn from_table(table: GpuTable, names: Vec<String>) -> Self {
        Self { table, names }
    }

    /// Create from individual columns and names.
    pub fn from_columns(columns: Vec<GpuColumn>, names: Vec<String>) -> PolarsResult<Self> {
        if columns.len() != names.len() {
            return Err(polars_err!(ComputeError:
                "GpuDataFrame: {} columns but {} names", columns.len(), names.len()));
        }
        let table = gpu_result(GpuTable::new(columns))?;
        Ok(Self { table, names })
    }

    /// Number of rows.
    pub fn height(&self) -> usize {
        self.table.num_rows()
    }

    /// Number of columns.
    pub fn width(&self) -> usize {
        self.table.num_columns()
    }

    /// Get a deep copy of the column at the given index.
    pub fn column(&self, index: usize) -> PolarsResult<GpuColumn> {
        gpu_result(self.table.column(index))
    }

    /// Find column index by name.
    pub fn column_index(&self, name: &str) -> PolarsResult<usize> {
        self.names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| polars_err!(ColumnNotFound: "{}", name))
    }

    /// Get a deep copy of the column with the given name.
    pub fn column_by_name(&self, name: &str) -> PolarsResult<GpuColumn> {
        let idx = self.column_index(name)?;
        self.column(idx)
    }

    /// Get the column names.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Select a subset of columns by name, producing a new GpuDataFrame.
    pub fn select_columns(&self, col_names: &[&str]) -> PolarsResult<Self> {
        let mut columns = Vec::with_capacity(col_names.len());
        let mut new_names = Vec::with_capacity(col_names.len());
        for &name in col_names {
            columns.push(self.column_by_name(name)?);
            new_names.push(name.to_string());
        }
        Self::from_columns(columns, new_names)
    }

    /// Apply a boolean mask, keeping only rows where the mask is true.
    pub fn apply_boolean_mask(&self, mask: &GpuColumn) -> PolarsResult<Self> {
        let filtered = gpu_result(self.table.apply_boolean_mask(mask))?;
        Ok(Self {
            table: filtered,
            names: self.names.clone(),
        })
    }

    /// Slice this frame: rows `[begin, begin+length)`.
    pub fn slice(&self, offset: i64, length: usize) -> PolarsResult<Self> {
        let height = self.height();
        let begin = if offset >= 0 {
            offset as usize
        } else {
            height.saturating_sub((-offset) as usize)
        };
        let end = (begin + length).min(height);
        let sliced = gpu_result(self.table.slice(begin, end))?;
        Ok(Self {
            table: sliced,
            names: self.names.clone(),
        })
    }

    /// Sort this frame by the given key columns.
    pub fn sort_by_key(
        &self,
        key_columns: Vec<GpuColumn>,
        orders: &[SortOrder],
        null_orders: &[NullOrder],
    ) -> PolarsResult<Self> {
        let keys_table = gpu_result(GpuTable::new(key_columns))?;
        let sorted = gpu_result(self.table.sort_by_key(&keys_table, orders, null_orders))?;
        Ok(Self {
            table: sorted,
            names: self.names.clone(),
        })
    }

    /// Perform a groupby aggregation.
    ///
    /// `key_columns` and `key_names` describe the groupby keys.
    /// `value_columns` are the columns to aggregate.
    /// `agg_kinds` maps each value column index (in `value_columns`) to its aggregation kind.
    /// `agg_names` are the output names for each aggregation result column.
    pub fn groupby(
        &self,
        key_columns: Vec<GpuColumn>,
        key_names: Vec<String>,
        value_columns: Vec<GpuColumn>,
        agg_requests: Vec<(usize, AggregationKind)>,
        agg_names: Vec<String>,
        maintain_order: bool,
    ) -> PolarsResult<Self> {
        let keys_table = gpu_result(GpuTable::new(key_columns))?;
        let values_table = gpu_result(GpuTable::new(value_columns))?;

        let mut gb = GroupBy::new(&keys_table);
        for (col_idx, kind) in agg_requests {
            gb = gb.agg(col_idx, kind);
        }

        let result = gpu_result(gb.execute(&values_table))?;

        // Result has key columns first, then aggregation columns
        let mut all_names = key_names.clone();
        all_names.extend(agg_names);

        let result_df = Self {
            table: result,
            names: all_names,
        };

        // If maintain_order is requested, sort by key columns to ensure deterministic output
        if maintain_order {
            let num_keys = key_names.len();
            let mut sort_keys = Vec::with_capacity(num_keys);
            for i in 0..num_keys {
                sort_keys.push(result_df.column(i)?);
            }
            let orders = vec![SortOrder::Ascending; num_keys];
            let null_orders = vec![NullOrder::After; num_keys];
            result_df.sort_by_key(sort_keys, &orders, &null_orders)
        } else {
            Ok(result_df)
        }
    }

    /// Remove duplicate rows based on the given subset of columns.
    pub fn distinct(
        &self,
        subset: Option<&[&str]>,
        keep: DuplicateKeepOption,
        maintain_order: bool,
    ) -> PolarsResult<Self> {
        let key_indices: Vec<usize> = match subset {
            Some(cols) => cols
                .iter()
                .map(|&name| self.column_index(name))
                .collect::<PolarsResult<_>>()?,
            None => (0..self.width()).collect(),
        };

        let result = if maintain_order {
            gpu_result(
                self.table
                    .stable_distinct(&key_indices, keep, NullEquality::Equal),
            )?
        } else {
            gpu_result(self.table.distinct(&key_indices, keep, NullEquality::Equal))?
        };

        Ok(Self {
            table: result,
            names: self.names.clone(),
        })
    }

    /// Access the inner cudf Table (for advanced operations).
    pub fn inner_table(&self) -> &GpuTable {
        &self.table
    }

    /// Convert back to a Polars DataFrame.
    pub fn to_polars(self) -> PolarsResult<DataFrame> {
        convert::gpu_to_dataframe(self.table, &self.names)
    }

    /// Create from a Polars DataFrame (upload to GPU).
    pub fn from_polars(df: &DataFrame) -> PolarsResult<Self> {
        let (table, names) = convert::dataframe_to_gpu(df)?;
        Ok(Self { table, names })
    }
}
