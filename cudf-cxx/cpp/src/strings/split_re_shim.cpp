#include "strings/split_re_shim.h"
#include <cudf/strings/split/split_re.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedTable> str_split_re(
    const OwnedColumn& col, rust::Str pattern, int32_t maxsplit)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string pat(pattern.data(), pattern.size());
    auto prog = cudf::strings::regex_program::create(pat);
    auto result = cudf::strings::split_re(col.view(), *prog, maxsplit, stream, mr);
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> str_rsplit_re(
    const OwnedColumn& col, rust::Str pattern, int32_t maxsplit)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string pat(pattern.data(), pattern.size());
    auto prog = cudf::strings::regex_program::create(pat);
    auto result = cudf::strings::rsplit_re(col.view(), *prog, maxsplit, stream, mr);
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_split_record_re(
    const OwnedColumn& col, rust::Str pattern, int32_t maxsplit)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string pat(pattern.data(), pattern.size());
    auto prog = cudf::strings::regex_program::create(pat);
    auto result = cudf::strings::split_record_re(col.view(), *prog, maxsplit, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_rsplit_record_re(
    const OwnedColumn& col, rust::Str pattern, int32_t maxsplit)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string pat(pattern.data(), pattern.size());
    auto prog = cudf::strings::regex_program::create(pat);
    auto result = cudf::strings::rsplit_record_re(col.view(), *prog, maxsplit, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
