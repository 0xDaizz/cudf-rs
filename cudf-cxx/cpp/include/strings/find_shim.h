#pragma once

#include <cudf/strings/find.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_find(
    const OwnedColumn& col, rust::Str target, int32_t start);
std::unique_ptr<OwnedColumn> str_rfind(
    const OwnedColumn& col, rust::Str target);
std::unique_ptr<OwnedColumn> str_starts_with(
    const OwnedColumn& col, rust::Str target);
std::unique_ptr<OwnedColumn> str_ends_with(
    const OwnedColumn& col, rust::Str target);

/// Check if each string contains any of the target strings.
/// Returns a table of BOOL8 columns, one per target.
std::unique_ptr<OwnedTable> str_contains_multiple(
    const OwnedColumn& col, const OwnedColumn& targets);

} // namespace cudf_shims
