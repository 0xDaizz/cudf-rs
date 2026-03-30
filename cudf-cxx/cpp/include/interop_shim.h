#pragma once

#include <cudf/interop.hpp>
#include <arrow/c/abi.h>
#include <cudf/io/types.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/contiguous_split.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

// ── Arrow IPC (legacy, retained for backward compat) ──────────

/// Export a column to Arrow IPC format (serialized bytes).
rust::Vec<uint8_t> column_to_arrow_ipc(const OwnedColumn& col);

/// Import a column from Arrow IPC format.
std::unique_ptr<OwnedColumn> column_from_arrow_ipc(rust::Slice<const uint8_t> data);

/// Export a table to Arrow IPC format.
rust::Vec<uint8_t> table_to_arrow_ipc(const OwnedTable& table);

/// Import a table from Arrow IPC format.
std::unique_ptr<OwnedTable> table_from_arrow_ipc(rust::Slice<const uint8_t> data);

// ── Arrow C Data Interface (true zero-copy) ───────────────────
//
// ArrowSchema / ArrowArray contain function pointers, which cxx cannot
// handle directly.  We heap-allocate them and pass raw pointers as u64.
// The Rust side uses arrow::ffi to import / export these structs.

/// Import a column from Arrow C Data Interface.
/// Takes ownership of schema + array (will be released internally).
std::unique_ptr<OwnedColumn> column_from_arrow_cdata(uint64_t schema_ptr, uint64_t array_ptr);

/// Import a table from Arrow C Data Interface.
std::unique_ptr<OwnedTable> table_from_arrow_cdata(uint64_t schema_ptr, uint64_t array_ptr);

/// Free an ArrowSchema / ArrowArray without consuming it (cleanup helper).
void free_arrow_schema(uint64_t ptr);
void free_arrow_array(uint64_t ptr);

// ── Arrow C Data Interface (paired export, single GPU→host copy) ─

/// Opaque pair holding both ArrowSchema and ArrowArray from a single export.
/// Avoids the double GPU→host copy that happens when calling
/// the separate schema/array export functions that were removed.
struct ArrowExportPair {
    ArrowSchema* schema;
    ArrowArray* array;

    ArrowExportPair() : schema(nullptr), array(nullptr) {}

    ~ArrowExportPair() {
        if (schema) { if (schema->release) schema->release(schema); delete schema; }
        if (array)  { if (array->release)  array->release(array);   delete array;  }
    }

    // Non-copyable.
    ArrowExportPair(const ArrowExportPair&) = delete;
    ArrowExportPair& operator=(const ArrowExportPair&) = delete;
};

/// Export a column's schema + array in a single GPU→host transfer.
std::unique_ptr<ArrowExportPair> column_to_arrow_pair(const OwnedColumn& col);

/// Export a table's schema + array in a single GPU→host transfer.
std::unique_ptr<ArrowExportPair> table_to_arrow_pair(const OwnedTable& table);

/// Take ownership of the schema pointer from the pair (returns u64, sets internal to null).
uint64_t arrow_pair_schema(ArrowExportPair& pair);

/// Take ownership of the array pointer from the pair (returns u64, sets internal to null).
uint64_t arrow_pair_array(ArrowExportPair& pair);

// ── DLPack ────────────────────────────────────────────────────
//
// DLManagedTensor* passed as u64 opaque handle.

/// Convert a table to DLPack tensor.  All columns must be numeric with
/// the same dtype and zero nulls.  Returns DLManagedTensor* as u64.
uint64_t table_to_dlpack(const OwnedTable& table);

/// Import a table from DLPack tensor.  Consumes the tensor.
std::unique_ptr<OwnedTable> table_from_dlpack(uint64_t dlpack_ptr);

/// Free a DLManagedTensor without consuming it.
void free_dlpack(uint64_t dlpack_ptr);

// ── contiguous_split / pack / unpack ──────────────────────────

/// Opaque wrapper around cudf::packed_columns.
struct OwnedPackedColumns {
    cudf::packed_columns inner;

    OwnedPackedColumns() = default;
    explicit OwnedPackedColumns(cudf::packed_columns&& pc)
        : inner(std::move(pc)) {}
};

/// Pack a table into a single contiguous GPU buffer + host metadata.
std::unique_ptr<OwnedPackedColumns> pack_table(const OwnedTable& table);

/// Get host-side metadata bytes from a packed table.
rust::Vec<uint8_t> packed_metadata(const OwnedPackedColumns& packed);

/// Get the size of the contiguous GPU data buffer (bytes).
int64_t packed_gpu_data_size(const OwnedPackedColumns& packed);

/// Unpack a packed table back into an OwnedTable.
/// (Performs a copy because the unpacked table_view borrows from packed.)
std::unique_ptr<OwnedTable> unpack_table(const OwnedPackedColumns& packed);

/// Perform contiguous_split on a table at the given split indices.
/// Returns a vector of packed tables (one per partition).
rust::Vec<uint64_t> contiguous_split_table(
    const OwnedTable& table,
    rust::Slice<const int32_t> splits);

/// Retrieve one packed table from a contiguous_split result by handle.
std::unique_ptr<OwnedPackedColumns> contiguous_split_get(uint64_t handle, int32_t index);

/// Free a contiguous_split result handle.
void contiguous_split_free(uint64_t handle);

} // namespace cudf_shims
