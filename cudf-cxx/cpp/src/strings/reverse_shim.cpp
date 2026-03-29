#include "strings/reverse_shim.h"
#include <cudf/strings/reverse.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_reverse(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::reverse(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
