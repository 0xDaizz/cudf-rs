//! Avro I/O (read-only).

use crate::error::{CudfError, Result};
use crate::table::Table;

pub struct AvroReader {
    path: String,
    columns: Vec<String>,
}

impl AvroReader {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            columns: vec![],
        }
    }
    pub fn columns(mut self, cols: Vec<String>) -> Self {
        self.columns = cols;
        self
    }
    pub fn read(self) -> Result<Table> {
        let raw = cudf_cxx::io::avro::ffi::read_avro(&self.path, &self.columns)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}

pub fn read_avro(path: impl Into<String>) -> Result<Table> {
    AvroReader::new(path).read()
}
