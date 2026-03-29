//! Avro I/O (read-only).

use crate::error::{CudfError, Result};
use crate::table::{Table, TableWithMetadata};

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

    /// Read an Avro file, returning both the table and column names
    /// from the file schema metadata.
    pub fn read_with_metadata(self) -> Result<TableWithMetadata> {
        let raw = cudf_cxx::io::avro::ffi::read_avro_with_metadata(&self.path, &self.columns)
            .map_err(CudfError::from_cxx)?;
        TableWithMetadata::from_raw(raw)
    }
}

pub fn read_avro(path: impl Into<String>) -> Result<Table> {
    AvroReader::new(path).read()
}
