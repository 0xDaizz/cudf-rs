#include "transpose_shim.h"
#include <cudf/transpose.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedTable> transpose_table(const OwnedTable& table) {
    auto [result_table, _] = cudf::transpose(table.view());
    return std::make_unique<OwnedTable>(std::move(result_table));
}

} // namespace cudf_shims
