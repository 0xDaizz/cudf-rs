#include "strings/repeat_shim.h"
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_repeat(
    const OwnedColumn& col, int32_t count)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::repeat_strings(
        col.view(), count, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
