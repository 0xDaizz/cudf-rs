#include "strings/replace_shim.h"
#include <cudf/strings/replace.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_replace(
    const OwnedColumn& col, rust::Str target, rust::Str replacement)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string tgt(target.data(), target.size());
    std::string repl(replacement.data(), replacement.size());
    cudf::string_scalar scalar_target(tgt, true, stream);
    cudf::string_scalar scalar_repl(repl, true, stream);
    auto result = cudf::strings::replace(
        col.view(), scalar_target, scalar_repl, -1, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_replace_re(
    const OwnedColumn& col, rust::Str pattern, rust::Str replacement)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string pat(pattern.data(), pattern.size());
    std::string repl(replacement.data(), replacement.size());
    auto prog = cudf::strings::regex_program::create(pat);
    cudf::string_scalar scalar_repl(repl, true, stream);
    auto result = cudf::strings::replace_re(
        col.view(), *prog, scalar_repl, std::nullopt, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
