#include "strings/findall_shim.h"
#include <cudf/strings/findall.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_findall(
    const OwnedColumn& col, rust::Str pattern)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string pat(pattern.data(), pattern.size());
    auto prog = cudf::strings::regex_program::create(pat);
    auto result = cudf::strings::findall(col.view(), *prog, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_find_re(
    const OwnedColumn& col, rust::Str pattern)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string pat(pattern.data(), pattern.size());
    auto prog = cudf::strings::regex_program::create(pat);
    auto result = cudf::strings::find_re(col.view(), *prog, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
