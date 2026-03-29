//! I/O operations for reading and writing GPU DataFrames.
//!
//! | Format | Read | Write |
//! |--------|------|-------|
//! | Parquet | `ParquetReader` | `ParquetWriter` |
//! | CSV | `CsvReader` | `CsvWriter` |
//! | JSON | `JsonReader` | `JsonWriter` |
//! | ORC | `OrcReader` | `OrcWriter` |
//! | Avro | `AvroReader` | - |

pub mod parquet;
pub mod csv;
pub mod json;
pub mod orc;
pub mod avro;
