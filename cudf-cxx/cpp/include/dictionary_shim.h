#pragma once

#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/types.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

/// Dictionary-encode a column.
std::unique_ptr<OwnedColumn> dictionary_encode(
    const OwnedColumn& col);

/// Decode a dictionary column back to its original representation.
std::unique_ptr<OwnedColumn> dictionary_decode(
    const OwnedColumn& col);

} // namespace cudf_shims
