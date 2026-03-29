#pragma once

#include <cudf/strings/split/split_re.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedTable> str_split_re(
    const OwnedColumn& col, rust::Str pattern, int32_t maxsplit);
std::unique_ptr<OwnedTable> str_rsplit_re(
    const OwnedColumn& col, rust::Str pattern, int32_t maxsplit);
std::unique_ptr<OwnedColumn> str_split_record_re(
    const OwnedColumn& col, rust::Str pattern, int32_t maxsplit);
std::unique_ptr<OwnedColumn> str_rsplit_record_re(
    const OwnedColumn& col, rust::Str pattern, int32_t maxsplit);

} // namespace cudf_shims
