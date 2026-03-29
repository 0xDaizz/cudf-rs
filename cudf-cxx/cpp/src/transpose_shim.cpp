#include "transpose_shim.h"
#include <cudf/transpose.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedTable> transpose_table(const OwnedTable& table) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto [owner_col, tv] = cudf::transpose(table.view(), stream, mr);
    // The table_view `tv` is backed by `owner_col`. We need to make
    // a deep copy into an owned cudf::table so it outlives this scope.
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(tv.num_columns());
    for (cudf::size_type i = 0; i < tv.num_columns(); ++i) {
        cols.push_back(std::make_unique<cudf::column>(tv.column(i), stream, mr));
    }
    auto result = std::make_unique<cudf::table>(std::move(cols));
    return std::make_unique<OwnedTable>(std::move(result));
}

} // namespace cudf_shims
