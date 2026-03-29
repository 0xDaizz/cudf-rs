//! Parquet I/O.

use crate::error::{CudfError, Result};
use crate::table::Table;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Compression { None = 0, Snappy = 1, Gzip = 2, Brotli = 3, Bz2 = 4, Lz4 = 5, Zstd = 6 }

pub struct ParquetReader { path: String, columns: Vec<String>, skip_rows: i64, num_rows: i64 }

impl ParquetReader {
    pub fn new(path: impl Into<String>) -> Self { Self { path: path.into(), columns: vec![], skip_rows: -1, num_rows: -1 } }
    pub fn columns(mut self, cols: Vec<String>) -> Self { self.columns = cols; self }
    pub fn skip_rows(mut self, n: usize) -> Self { self.skip_rows = n as i64; self }
    pub fn num_rows(mut self, n: usize) -> Self { self.num_rows = n as i64; self }
    pub fn read(self) -> Result<Table> {
        let raw = cudf_cxx::io::parquet::ffi::read_parquet(&self.path, &self.columns, self.skip_rows, self.num_rows).map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}

pub struct ParquetWriter<'a> { table: &'a Table, path: String, compression: Compression }

impl<'a> ParquetWriter<'a> {
    pub fn new(table: &'a Table, path: impl Into<String>) -> Self { Self { table, path: path.into(), compression: Compression::Snappy } }
    pub fn compression(mut self, c: Compression) -> Self { self.compression = c; self }
    pub fn write(self) -> Result<()> { cudf_cxx::io::parquet::ffi::write_parquet(&self.table.inner, &self.path, self.compression as i32).map_err(CudfError::from_cxx) }
}

pub fn read_parquet(path: impl Into<String>) -> Result<Table> { ParquetReader::new(path).read() }
pub fn write_parquet(table: &Table, path: impl Into<String>) -> Result<()> { ParquetWriter::new(table, path).write() }
