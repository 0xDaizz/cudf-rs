#pragma once

#include <cudf/join.hpp>
#include <memory>
#include "rust/cxx.h"
#include "table_shim.h"
#include "column_shim.h"

namespace cudf_shims {

// Inner/left/full return a 2-column table of [left_indices, right_indices].
std::unique_ptr<OwnedTable> inner_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys);
std::unique_ptr<OwnedTable> left_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys);
std::unique_ptr<OwnedTable> full_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys);

// Semi/anti return a single column of indices.
std::unique_ptr<OwnedColumn> left_semi_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys);
std::unique_ptr<OwnedColumn> left_anti_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys);

// Cross join returns the full cartesian product table.
std::unique_ptr<OwnedTable> cross_join(
    const OwnedTable& left, const OwnedTable& right);

} // namespace cudf_shims
