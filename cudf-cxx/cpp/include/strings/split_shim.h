#pragma once

#include <cudf/strings/split/split.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedTable> str_split(
    const OwnedColumn& col, rust::Str delimiter, int32_t maxsplit);
std::unique_ptr<OwnedTable> str_rsplit(
    const OwnedColumn& col, rust::Str delimiter, int32_t maxsplit);

/// Split each string into a list column of strings.
std::unique_ptr<OwnedColumn> str_split_record(
    const OwnedColumn& col, rust::Str delimiter, int32_t maxsplit);
std::unique_ptr<OwnedColumn> str_rsplit_record(
    const OwnedColumn& col, rust::Str delimiter, int32_t maxsplit);

} // namespace cudf_shims
