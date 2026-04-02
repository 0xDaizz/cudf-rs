#include "round_shim.h"
#include <cudf/round.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> round_column(const OwnedColumn& col, int32_t decimal_places) {
    auto result = cudf::round_decimal(
        col.view(),
        decimal_places,
        cudf::rounding_method::HALF_UP);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
