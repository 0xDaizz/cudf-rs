//! CSV I/O.

use crate::error::{CudfError, Result};
use crate::table::Table;

pub struct CsvReader {
    path: String,
    delimiter: u8,
    header_row: i32,
    skip_rows: i64,
    num_rows: i64,
}

impl CsvReader {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            delimiter: b',',
            header_row: 0,
            skip_rows: -1,
            num_rows: -1,
        }
    }
    pub fn delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }
    pub fn no_header(mut self) -> Self {
        self.header_row = -1;
        self
    }
    pub fn skip_rows(mut self, n: usize) -> Self {
        self.skip_rows = n as i64;
        self
    }
    pub fn num_rows(mut self, n: usize) -> Self {
        self.num_rows = n as i64;
        self
    }
    pub fn read(self) -> Result<Table> {
        let raw = cudf_cxx::io::csv::ffi::read_csv(
            &self.path,
            self.delimiter,
            self.header_row,
            self.skip_rows,
            self.num_rows,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}

pub struct CsvWriter<'a> {
    table: &'a Table,
    path: String,
    delimiter: u8,
    header: bool,
}

impl<'a> CsvWriter<'a> {
    pub fn new(table: &'a Table, path: impl Into<String>) -> Self {
        Self {
            table,
            path: path.into(),
            delimiter: b',',
            header: true,
        }
    }
    pub fn delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }
    pub fn no_header(mut self) -> Self {
        self.header = false;
        self
    }
    pub fn write(self) -> Result<()> {
        cudf_cxx::io::csv::ffi::write_csv(
            &self.table.inner,
            &self.path,
            self.delimiter,
            self.header,
        )
        .map_err(CudfError::from_cxx)
    }
}

pub fn read_csv(path: impl Into<String>) -> Result<Table> {
    CsvReader::new(path).read()
}
pub fn write_csv(table: &Table, path: impl Into<String>) -> Result<()> {
    CsvWriter::new(table, path).write()
}
