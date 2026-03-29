#pragma once

#include <cudf/json/json.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

/// Apply a JSONPath query to each string in the column.
std::unique_ptr<OwnedColumn> get_json_object(
    const OwnedColumn& col,
    rust::Str json_path,
    bool allow_single_quotes,
    bool strip_quotes,
    bool missing_fields_as_nulls);

} // namespace cudf_shims
