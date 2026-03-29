#include "label_bins_shim.h"
#include <cudf/labeling/label_bins.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> label_bins(
    const OwnedColumn& input,
    const OwnedColumn& left_edges,
    bool left_inclusive,
    const OwnedColumn& right_edges,
    bool right_inclusive)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::label_bins(
        input.view(),
        left_edges.view(),
        left_inclusive ? cudf::inclusive::YES : cudf::inclusive::NO,
        right_edges.view(),
        right_inclusive ? cudf::inclusive::YES : cudf::inclusive::NO,
        stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
