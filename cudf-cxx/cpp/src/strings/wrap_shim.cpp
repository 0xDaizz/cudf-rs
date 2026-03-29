#include "strings/wrap_shim.h"
#include <cudf/strings/wrap.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_wrap(const OwnedColumn& col, int32_t width) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::wrap(col.view(), width, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
