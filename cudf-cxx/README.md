# cudf-cxx

Low-level [cxx](https://cxx.rs)-based FFI bridge between Rust and NVIDIA's libcudf C++ API.

> **Note:** Most users should use the high-level [`cudf`](../cudf/) crate instead.

## Architecture

```
cudf-cxx/
├── src/           # Rust cxx::bridge modules (one per libcudf module)
│   ├── types.rs   # type_id, data_type
│   ├── column.rs  # column, column_view
│   ├── table.rs   # table, table_view
│   ├── io/        # parquet, csv, json, orc, avro
│   ├── strings/   # case, combine, contains, convert, extract, find, replace, slice, split, strip
│   └── ...        # 37 top-level modules total
└── cpp/           # C++ shim layer
    ├── include/   # Header files (.h)
    └── src/       # Implementation files (.cpp)
```

Each libcudf module has three corresponding files:

| Layer | File | Purpose |
|-------|------|---------|
| C++ shim | `cpp/include/X_shim.h` + `cpp/src/X_shim.cpp` | Wraps libcudf types into cxx-compatible opaque structs |
| cxx bridge | `src/X.rs` | Declares the FFI interface via `#[cxx::bridge]` |

## Complete Module List

### Core (3)

| Bridge | C++ Shim | libcudf Module |
|--------|----------|----------------|
| `types.rs` | `types_shim` | `cudf/types.hpp` |
| `column.rs` | `column_shim` | `cudf/column/` |
| `table.rs` | `table_shim` | `cudf/table/` |

### Compute (15)

| Bridge | C++ Shim | libcudf Module |
|--------|----------|----------------|
| `sorting.rs` | `sorting_shim` | `cudf/sorting/` |
| `groupby.rs` | `groupby_shim` | `cudf/groupby/` |
| `aggregation.rs` | `aggregation_shim` | `cudf/aggregation.hpp` |
| `reduction.rs` | `reduction_shim` | `cudf/reduction/` |
| `quantiles.rs` | `quantiles_shim` | `cudf/quantiles/` |
| `rolling.rs` | `rolling_shim` | `cudf/rolling/` |
| `binaryop.rs` | `binaryop_shim` | `cudf/binaryop/` |
| `unary.rs` | `unary_shim` | `cudf/unary/` |
| `round.rs` | `round_shim` | `cudf/round.hpp` |
| `transform.rs` | `transform_shim` | `cudf/transform.hpp` |
| `search.rs` | `search_shim` | `cudf/search/` |
| `hashing.rs` | `hashing_shim` | `cudf/hashing/` |
| `datetime.rs` | `datetime_shim` | `cudf/datetime/` |
| `scalar.rs` | `scalar_shim` | `cudf/scalar/` |
| `label_bins.rs` | `label_bins_shim` | `cudf/labeling/` |

### Data Manipulation (16)

| Bridge | C++ Shim | libcudf Module |
|--------|----------|----------------|
| `copying.rs` | `copying_shim` | `cudf/copying.hpp` |
| `filling.rs` | `filling_shim` | `cudf/filling.hpp` |
| `concatenate.rs` | `concatenate_shim` | `cudf/concatenate.hpp` |
| `merge.rs` | `merge_shim` | `cudf/merge.hpp` |
| `join.rs` | `join_shim` | `cudf/join/` |
| `stream_compaction.rs` | `stream_compaction_shim` | `cudf/stream_compaction.hpp` |
| `null_mask.rs` | `null_mask_shim` | `cudf/null_mask.hpp` |
| `reshape.rs` | `reshape_shim` | `cudf/reshape.hpp` |
| `transpose.rs` | `transpose_shim` | `cudf/transpose.hpp` |
| `partitioning.rs` | `partitioning_shim` | `cudf/partitioning.hpp` |
| `replace.rs` | `replace_shim` | `cudf/replace.hpp` |
| `dictionary.rs` | `dictionary_shim` | `cudf/dictionary/` |
| `json.rs` | `json_shim` | `cudf/json/` |
| `lists/ops.rs` | `lists/lists_shim` | `cudf/lists/` |
| `structs.rs` | `structs_shim` | `cudf/structs/` |
| `timezone.rs` | `timezone_shim` | `cudf/io/timezone.hpp` |

### I/O (5)

| Bridge | C++ Shim | libcudf Module |
|--------|----------|----------------|
| `io/parquet.rs` | `io/parquet_shim` | `cudf/io/parquet.hpp` |
| `io/csv.rs` | `io/csv_shim` | `cudf/io/csv.hpp` |
| `io/json.rs` | `io/json_shim` | `cudf/io/json.hpp` |
| `io/orc.rs` | `io/orc_shim` | `cudf/io/orc.hpp` |
| `io/avro.rs` | `io/avro_shim` | `cudf/io/avro.hpp` |

### String Operations (21)

| Bridge | C++ Shim | libcudf Module |
|--------|----------|----------------|
| `strings/case.rs` | `strings/case_shim` | `cudf/strings/case.hpp` |
| `strings/combine.rs` | `strings/combine_shim` | `cudf/strings/combine.hpp` |
| `strings/contains.rs` | `strings/contains_shim` | `cudf/strings/contains.hpp` |
| `strings/convert.rs` | `strings/convert_shim` | `cudf/strings/convert/` |
| `strings/extract.rs` | `strings/extract_shim` | `cudf/strings/extract.hpp` |
| `strings/find.rs` | `strings/find_shim` | `cudf/strings/find.hpp` |
| `strings/replace.rs` | `strings/replace_shim` | `cudf/strings/replace.hpp` |
| `strings/slice.rs` | `strings/slice_shim` | `cudf/strings/substring.hpp` |
| `strings/split.rs` | `strings/split_shim` | `cudf/strings/split/` |
| `strings/strip.rs` | `strings/strip_shim` | `cudf/strings/strip.hpp` |
| `strings/split_re.rs` | `strings/split_re_shim` | `cudf/strings/split/` |
| `strings/partition.rs` | `strings/partition_shim` | `cudf/strings/split/partition.hpp` |
| `strings/padding.rs` | `strings/padding_shim` | `cudf/strings/padding.hpp` |
| `strings/repeat.rs` | `strings/repeat_shim` | `cudf/strings/repeat_strings.hpp` |
| `strings/findall.rs` | `strings/findall_shim` | `cudf/strings/findall.hpp` |
| `strings/attributes.rs` | `strings/attributes_shim` | `cudf/strings/attributes.hpp` |
| `strings/translate.rs` | `strings/translate_shim` | `cudf/strings/translate.hpp` |
| `strings/reverse.rs` | `strings/reverse_shim` | `cudf/strings/reverse.hpp` |
| `strings/wrap.rs` | `strings/wrap_shim` | `cudf/strings/wrap.hpp` |
| `strings/char_types.rs` | `strings/char_types_shim` | `cudf/strings/char_types.hpp` |
| `strings/like.rs` | `strings/like_shim` | `cudf/strings/like.hpp` |

### Interop (1)

| Bridge | C++ Shim | libcudf Module |
|--------|----------|----------------|
| `interop.rs` | `interop_shim` | `cudf/interop.hpp` |

**Total: 61 modules** (3 core + 15 compute + 16 data manipulation + 5 I/O + 21 strings + 1 interop)

## Adding a New Binding

1. Create `cpp/include/new_module_shim.h` -- define the shim struct and function signatures
2. Create `cpp/src/new_module_shim.cpp` -- implement the shim functions
3. Create `src/new_module.rs` -- write the `#[cxx::bridge]` declarations
4. Add the bridge file to `build.rs`'s `cxx_build::bridges()` list
5. Add `pub mod new_module;` to `src/lib.rs`

## Design Principles

- **One OwnedX struct per libcudf owning type**: `OwnedColumn` wraps `unique_ptr<column>`, `OwnedTable` wraps `unique_ptr<table>`.
- **Flat parameter passing**: Options/config structs do NOT cross FFI. The shim constructs them from flat parameters.
- **All functions returning Result**: Any C++ function that can throw is declared with `-> Result<T>` in the bridge, so cxx automatically converts exceptions.
- **No templates in bridge**: Template instantiations are handled in the C++ shim layer.
