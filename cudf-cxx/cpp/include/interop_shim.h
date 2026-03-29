#pragma once

#include <cudf/interop.hpp>
#include <cudf/io/types.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

/// Export a column to Arrow IPC format (serialized bytes).
rust::Vec<uint8_t> column_to_arrow_ipc(const OwnedColumn& col);

/// Import a column from Arrow IPC format.
std::unique_ptr<OwnedColumn> column_from_arrow_ipc(rust::Slice<const uint8_t> data);

/// Export a table to Arrow IPC format.
rust::Vec<uint8_t> table_to_arrow_ipc(const OwnedTable& table);

/// Import a table from Arrow IPC format.
std::unique_ptr<OwnedTable> table_from_arrow_ipc(rust::Slice<const uint8_t> data);

} // namespace cudf_shims
