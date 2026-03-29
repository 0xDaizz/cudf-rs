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

/// Find positions of multiple target strings in each string.
/// Returns a lists column of INT32 positions.
std::unique_ptr<OwnedColumn> str_find_multiple(
    const OwnedColumn& col, const OwnedColumn& targets);

/// Find targets (column) in each string, returning positions.
std::unique_ptr<OwnedColumn> str_find_column(
    const OwnedColumn& col, const OwnedColumn& targets, int32_t start);

/// Find the nth instance of target in each string.
std::unique_ptr<OwnedColumn> str_find_instance(
    const OwnedColumn& col, rust::Str target, int32_t instance);

/// Check if each string contains the corresponding target string (column-based).
std::unique_ptr<OwnedColumn> str_contains_column(
    const OwnedColumn& col, const OwnedColumn& targets);

/// Check if each string starts with the corresponding target string (column-based).
std::unique_ptr<OwnedColumn> str_starts_with_column(
    const OwnedColumn& col, const OwnedColumn& targets);

/// Check if each string ends with the corresponding target string (column-based).
std::unique_ptr<OwnedColumn> str_ends_with_column(
    const OwnedColumn& col, const OwnedColumn& targets);

} // namespace cudf_shims
