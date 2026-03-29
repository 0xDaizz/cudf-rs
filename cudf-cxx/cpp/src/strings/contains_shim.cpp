#include "strings/contains_shim.h"
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_contains(
    const OwnedColumn& col, rust::Str target)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string tgt(target.data(), target.size());
    cudf::string_scalar scalar_target(tgt, true, stream);
    auto result = cudf::strings::contains(col.view(), scalar_target, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_contains_re(
    const OwnedColumn& col, rust::Str pattern)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string pat(pattern.data(), pattern.size());
    auto prog = cudf::strings::regex_program::create(pat);
    auto result = cudf::strings::contains_re(col.view(), *prog, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_matches_re(
    const OwnedColumn& col, rust::Str pattern)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string pat(pattern.data(), pattern.size());
    auto prog = cudf::strings::regex_program::create(pat);
    auto result = cudf::strings::matches_re(col.view(), *prog, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_count_re(
    const OwnedColumn& col, rust::Str pattern)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string pat(pattern.data(), pattern.size());
    auto prog = cudf::strings::regex_program::create(pat);
    auto result = cudf::strings::count_re(col.view(), *prog, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
