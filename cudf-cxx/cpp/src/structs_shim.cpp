#include "structs_shim.h"
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <stdexcept>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> structs_extract(
    const OwnedColumn& col,
    int32_t index)
{
    auto structs_view = cudf::structs_column_view(col.view());

    if (index < 0 || index >= structs_view.num_children()) {
        throw std::out_of_range(
            "structs_extract: index " + std::to_string(index) +
            " out of range [0, " + std::to_string(structs_view.num_children()) + ")");
    }

    auto child_view = structs_view.get_sliced_child(index);
    // Materialize a copy so the returned column owns its own memory.
    auto result = std::make_unique<cudf::column>(child_view);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
