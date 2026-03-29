#pragma once

#include <cudf/reshape.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

/// Interleave columns of a table into a single column.
std::unique_ptr<OwnedColumn> interleave_columns(const OwnedTable& table);

/// Repeat (tile) a table's rows `count` times.
std::unique_ptr<OwnedTable> tile(const OwnedTable& table, int32_t count);

/// Convert a column's elements to lists of bytes.
/// `flip_endianness`: false=NO, true=YES.
std::unique_ptr<OwnedColumn> byte_cast(
    const OwnedColumn& col, bool flip_endianness);

} // namespace cudf_shims
