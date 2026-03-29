//! ORC I/O.

use crate::error::{CudfError, Result};
use crate::table::Table;
use crate::io::parquet::Compression;

pub struct OrcReader { path: String, columns: Vec<String>, skip_rows: i64, num_rows: i64 }

impl OrcReader {
    pub fn new(path: impl Into<String>) -> Self { Self { path: path.into(), columns: vec![], skip_rows: -1, num_rows: -1 } }
    pub fn columns(mut self, cols: Vec<String>) -> Self { self.columns = cols; self }
    pub fn skip_rows(mut self, n: usize) -> Self { self.skip_rows = n as i64; self }
    pub fn num_rows(mut self, n: usize) -> Self { self.num_rows = n as i64; self }
    pub fn read(self) -> Result<Table> {
        let raw = cudf_cxx::io::orc::ffi::read_orc(&self.path, &self.columns, self.skip_rows, self.num_rows).map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}

pub struct OrcWriter<'a> { table: &'a Table, path: String, compression: Compression }

impl<'a> OrcWriter<'a> {
    pub fn new(table: &'a Table, path: impl Into<String>) -> Self { Self { table, path: path.into(), compression: Compression::Snappy } }
    pub fn compression(mut self, c: Compression) -> Self { self.compression = c; self }
    pub fn write(self) -> Result<()> { cudf_cxx::io::orc::ffi::write_orc(&self.table.inner, &self.path, self.compression as i32).map_err(CudfError::from_cxx) }
}

pub fn read_orc(path: impl Into<String>) -> Result<Table> { OrcReader::new(path).read() }
pub fn write_orc(table: &Table, path: impl Into<String>) -> Result<()> { OrcWriter::new(table, path).write() }
