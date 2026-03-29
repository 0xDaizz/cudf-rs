#pragma once

#include <cudf/structs/structs_column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

/// Extract the child column at index from a struct column.
/// Returns a materialized copy of the child.
std::unique_ptr<OwnedColumn> structs_extract(
    const OwnedColumn& col,
    int32_t index);

} // namespace cudf_shims
