#include "strings/slice_shim.h"
#include <cudf/strings/slice.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_slice(
    const OwnedColumn& col, int32_t start, int32_t stop)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    cudf::numeric_scalar<int32_t> scalar_start(start, true, stream);
    cudf::numeric_scalar<int32_t> scalar_stop(stop, true, stream);
    cudf::numeric_scalar<int32_t> scalar_step(1, true, stream);
    auto result = cudf::strings::slice_strings(
        col.view(), scalar_start, scalar_stop, scalar_step, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
