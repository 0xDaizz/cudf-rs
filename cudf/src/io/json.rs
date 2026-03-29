//! JSON I/O.

use crate::error::{CudfError, Result};
use crate::table::{Table, TableWithMetadata};

pub struct JsonReader {
    path: String,
    lines: bool,
}

impl JsonReader {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            lines: false,
        }
    }
    pub fn lines(mut self, l: bool) -> Self {
        self.lines = l;
        self
    }
    pub fn read(self) -> Result<Table> {
        let raw = cudf_cxx::io::json::ffi::read_json(&self.path, self.lines)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }

    /// Read a JSON file, returning both the table and column names
    /// from the JSON keys.
    pub fn read_with_metadata(self) -> Result<TableWithMetadata> {
        let raw = cudf_cxx::io::json::ffi::read_json_with_metadata(&self.path, self.lines)
            .map_err(CudfError::from_cxx)?;
        TableWithMetadata::from_raw(raw)
    }
}

pub struct JsonWriter<'a> {
    table: &'a Table,
    path: String,
    lines: bool,
}

impl<'a> JsonWriter<'a> {
    pub fn new(table: &'a Table, path: impl Into<String>) -> Self {
        Self {
            table,
            path: path.into(),
            lines: false,
        }
    }
    pub fn lines(mut self, l: bool) -> Self {
        self.lines = l;
        self
    }
    pub fn write(self) -> Result<()> {
        cudf_cxx::io::json::ffi::write_json(&self.table.inner, &self.path, self.lines)
            .map_err(CudfError::from_cxx)
    }
}

pub fn read_json(path: impl Into<String>) -> Result<Table> {
    JsonReader::new(path).read()
}
pub fn write_json(table: &Table, path: impl Into<String>) -> Result<()> {
    JsonWriter::new(table, path).write()
}
