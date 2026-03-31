//! GpuDataFrame: a named collection of GPU-resident columns.
//!
//! Wraps a `cudf::Table` with column names, providing named access
//! and conversion back to Polars `DataFrame`.

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
