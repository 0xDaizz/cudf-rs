#pragma once

#include <cudf/strings/extract.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedTable> str_extract(
    const OwnedColumn& col, rust::Str pattern);

/// Extract all matches of capture groups per row, returning a list column.
std::unique_ptr<OwnedColumn> str_extract_all_record(
    const OwnedColumn& col, rust::Str pattern);

/// Extract a single capture group from each string.
std::unique_ptr<OwnedColumn> str_extract_single(
    const OwnedColumn& col, rust::Str pattern, int32_t group_index);

} // namespace cudf_shims
