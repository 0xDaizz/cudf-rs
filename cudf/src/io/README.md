# I/O Module

GPU-accelerated file I/O for reading and writing tabular data directly to/from GPU memory.

All readers and writers use a builder pattern for configuration, plus convenience functions for one-liner usage.

## Supported Formats

| Format | Read | Write | Module |
|--------|------|-------|--------|
| Parquet | `ParquetReader` | `ParquetWriter` | `io::parquet` |
| CSV | `CsvReader` | `CsvWriter` | `io::csv` |
| JSON | `JsonReader` | `JsonWriter` | `io::json` |
| ORC | `OrcReader` | `OrcWriter` | `io::orc` |
| Avro | `AvroReader` | -- | `io::avro` |

## Examples

### Parquet

```rust,no_run
use cudf::io::parquet::{self, ParquetReader, ParquetWriter, Compression};
use cudf::Table;

// Quick read
let table = parquet::read_parquet("data.parquet")?;

// Builder: select columns, skip rows
let table = ParquetReader::new("data.parquet")
    .columns(vec!["id".into(), "value".into()])
    .skip_rows(100)
    .num_rows(1000)
    .read()?;

// Write with Zstd compression
ParquetWriter::new(&table, "output.parquet")
    .compression(Compression::Zstd)
    .write()?;
```

### CSV

```rust,no_run
use cudf::io::csv::{self, CsvReader, CsvWriter};

// Quick read
let table = csv::read_csv("data.csv")?;

// Builder: TSV, no header, skip first 10 rows
let table = CsvReader::new("data.tsv")
    .delimiter(b'\t')
    .no_header()
    .skip_rows(10)
    .read()?;

// Write without header
CsvWriter::new(&table, "output.csv")
    .no_header()
    .write()?;
```

### JSON

```rust,no_run
use cudf::io::json::{self, JsonReader, JsonWriter};

// Standard JSON
let table = json::read_json("data.json")?;

// JSON Lines (newline-delimited)
let table = JsonReader::new("data.jsonl")
    .lines(true)
    .read()?;

// Write as JSON Lines
JsonWriter::new(&table, "output.jsonl")
    .lines(true)
    .write()?;
```

### ORC

```rust,no_run
use cudf::io::orc::{self, OrcReader, OrcWriter};
use cudf::io::parquet::Compression;

// Quick read
let table = orc::read_orc("data.orc")?;

// Builder: select columns
let table = OrcReader::new("data.orc")
    .columns(vec!["col_a".into(), "col_b".into()])
    .num_rows(5000)
    .read()?;

// Write with Snappy compression
OrcWriter::new(&table, "output.orc")
    .compression(Compression::Snappy)
    .write()?;
```

### Avro (read-only)

```rust,no_run
use cudf::io::avro::{self, AvroReader};

// Quick read
let table = avro::read_avro("data.avro")?;

// Select columns
let table = AvroReader::new("data.avro")
    .columns(vec!["field_a".into(), "field_b".into()])
    .read()?;
```

## Compression Options (Parquet/ORC)

| Variant | Value | Description |
|---------|-------|-------------|
| `Compression::None` | 0 | No compression |
| `Compression::Auto` | 1 | Auto-detect |
| `Compression::Snappy` | 2 | Snappy (default) |
| `Compression::Gzip` | 3 | Gzip |
| `Compression::Bzip2` | 4 | Bzip2 |
| `Compression::Brotli` | 5 | Brotli |
| `Compression::Zip` | 6 | Zip |
| `Compression::Xz` | 7 | Xz |
| `Compression::Zlib` | 8 | Zlib |
| `Compression::Lz4` | 9 | LZ4 |
| `Compression::Lzo` | 10 | LZO |
| `Compression::Zstd` | 11 | Zstandard |
