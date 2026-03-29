//! Parquet I/O.

use crate::error::{CudfError, Result};
use crate::table::{Table, TableWithMetadata};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Compression {
    None = 0,
    Auto = 1,
    Snappy = 2,
    Gzip = 3,
    Bzip2 = 4,
    Brotli = 5,
    Zip = 6,
    Xz = 7,
    Zlib = 8,
    Lz4 = 9,
    Lzo = 10,
    Zstd = 11,
}

pub struct ParquetReader {
    path: String,
    columns: Vec<String>,
    skip_rows: i64,
    num_rows: i64,
}

impl ParquetReader {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            columns: vec![],
            skip_rows: -1,
            num_rows: -1,
        }
    }
    pub fn columns(mut self, cols: Vec<String>) -> Self {
        self.columns = cols;
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
        let raw = cudf_cxx::io::parquet::ffi::read_parquet(
            &self.path,
            &self.columns,
            self.skip_rows,
            self.num_rows,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }

    /// Read a Parquet file, returning both the table and column names
    /// from the file schema metadata.
    pub fn read_with_metadata(self) -> Result<TableWithMetadata> {
        let raw = cudf_cxx::io::parquet::ffi::read_parquet_with_metadata(
            &self.path,
            &self.columns,
            self.skip_rows,
            self.num_rows,
        )
        .map_err(CudfError::from_cxx)?;
        TableWithMetadata::from_raw(raw)
    }
}

pub struct ParquetWriter<'a> {
    table: &'a Table,
    path: String,
    compression: Compression,
}

impl<'a> ParquetWriter<'a> {
    pub fn new(table: &'a Table, path: impl Into<String>) -> Self {
        Self {
            table,
            path: path.into(),
            compression: Compression::Snappy,
        }
    }

    pub fn compression(mut self, c: Compression) -> Self {
        self.compression = c;
        self
    }
    pub fn write(self) -> Result<()> {
        cudf_cxx::io::parquet::ffi::write_parquet(
            &self.table.inner,
            &self.path,
            self.compression as i32,
        )
        .map_err(CudfError::from_cxx)
    }
}

pub fn read_parquet(path: impl Into<String>) -> Result<Table> {
    ParquetReader::new(path).read()
}
pub fn write_parquet(table: &Table, path: impl Into<String>) -> Result<()> {
    ParquetWriter::new(table, path).write()
}

// ── Parquet Metadata ─────────────────────────────────────────

/// Metadata about a Parquet file, read without loading the data.
pub struct ParquetMetadata {
    inner: cxx::UniquePtr<cudf_cxx::io::parquet::ffi::OwnedParquetMetadata>,
}

impl ParquetMetadata {
    /// Number of rows in the file.
    pub fn num_rows(&self) -> i64 {
        cudf_cxx::io::parquet::ffi::get_num_rows(&self.inner)
    }

    /// Number of row groups in the file.
    pub fn num_row_groups(&self) -> i32 {
        cudf_cxx::io::parquet::ffi::get_num_row_groups(&self.inner)
    }

    /// Number of columns in the file.
    pub fn num_columns(&self) -> i32 {
        cudf_cxx::io::parquet::ffi::get_num_columns(&self.inner)
    }

    /// Get column names from the schema.
    pub fn column_names(&self) -> Vec<String> {
        let n = self.num_columns();
        (0..n)
            .map(|i| cudf_cxx::io::parquet::ffi::get_column_name(&self.inner, i))
            .collect()
    }
}

/// Read metadata from a Parquet file without loading the data.
///
/// Returns file-level information including number of rows,
/// number of row groups, and column names from the schema.
pub fn read_parquet_metadata(path: impl Into<String>) -> Result<ParquetMetadata> {
    let path = path.into();
    let inner =
        cudf_cxx::io::parquet::ffi::read_parquet_metadata(&path).map_err(CudfError::from_cxx)?;
    Ok(ParquetMetadata { inner })
}

// ── Chunked Parquet Reader ────────────────────────────────────

/// A chunked Parquet reader for reading large files in pieces.
///
/// This reader is useful when a Parquet file is too large to read into
/// memory at once. It reads the file chunk by chunk, where each chunk
/// is bounded by `chunk_read_limit` bytes.
///
/// # Examples
///
/// ```rust,no_run
/// use cudf::io::parquet::ChunkedParquetReader;
///
/// let mut reader = ChunkedParquetReader::new("large_file.parquet", 256 * 1024 * 1024).unwrap();
/// while reader.has_next().unwrap() {
///     let chunk = reader.read_chunk().unwrap();
///     // process chunk...
/// }
/// ```
pub struct ChunkedParquetReader {
    inner: cxx::UniquePtr<cudf_cxx::io::parquet::ffi::OwnedChunkedParquetReader>,
}

// SAFETY: GPU memory is process-global.
unsafe impl Send for ChunkedParquetReader {}

impl ChunkedParquetReader {
    /// Create a chunked Parquet reader.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Parquet file.
    /// * `chunk_read_limit` - Maximum number of bytes per chunk (0 = no limit).
    pub fn new(path: impl Into<String>, chunk_read_limit: usize) -> Result<Self> {
        let path = path.into();
        let inner = cudf_cxx::io::parquet::ffi::chunked_parquet_reader_create(
            &path,
            chunk_read_limit as i64,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Self { inner })
    }

    /// Check if there is more data to read.
    pub fn has_next(&self) -> Result<bool> {
        cudf_cxx::io::parquet::ffi::chunked_parquet_reader_has_next(&self.inner)
            .map_err(CudfError::from_cxx)
    }

    /// Read the next chunk as a [`Table`].
    ///
    /// Returns an empty table if all data has been read.
    pub fn read_chunk(&self) -> Result<Table> {
        let raw = cudf_cxx::io::parquet::ffi::chunked_parquet_reader_read_chunk(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}

// ── Chunked Parquet Writer ────────────────────────────────────

/// A chunked Parquet writer for writing large tables in pieces.
///
/// This writer allows writing a table incrementally, chunk by chunk,
/// to a single Parquet file. Call [`write`] for each chunk and
/// [`close`] when done.
///
/// # Examples
///
/// ```rust,no_run
/// use cudf::{Column, Table};
/// use cudf::io::parquet::ChunkedParquetWriter;
///
/// let mut writer = ChunkedParquetWriter::new("output.parquet").unwrap();
///
/// for _ in 0..10 {
///     let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
///     let chunk = Table::new(vec![col]).unwrap();
///     writer.write(&chunk).unwrap();
/// }
///
/// writer.close().unwrap();
/// ```
pub struct ChunkedParquetWriter {
    inner: cxx::UniquePtr<cudf_cxx::io::parquet::ffi::OwnedChunkedParquetWriter>,
}

// SAFETY: GPU memory is process-global.
unsafe impl Send for ChunkedParquetWriter {}

impl ChunkedParquetWriter {
    /// Create a chunked Parquet writer with default Snappy compression.
    pub fn new(path: impl Into<String>) -> Result<Self> {
        Self::with_compression(path, Compression::Snappy)
    }

    /// Create a chunked Parquet writer with the specified compression.
    pub fn with_compression(path: impl Into<String>, compression: Compression) -> Result<Self> {
        let path = path.into();
        let inner =
            cudf_cxx::io::parquet::ffi::chunked_parquet_writer_create(&path, compression as i32)
                .map_err(CudfError::from_cxx)?;
        Ok(Self { inner })
    }

    /// Write a table chunk to the Parquet file.
    pub fn write(&mut self, table: &Table) -> Result<()> {
        cudf_cxx::io::parquet::ffi::chunked_parquet_writer_write(self.inner.pin_mut(), &table.inner)
            .map_err(CudfError::from_cxx)
    }

    /// Finalize and close the Parquet file.
    ///
    /// This must be called to ensure all data is flushed.
    pub fn close(&mut self) -> Result<()> {
        cudf_cxx::io::parquet::ffi::chunked_parquet_writer_close(self.inner.pin_mut())
            .map_err(CudfError::from_cxx)
    }
}
