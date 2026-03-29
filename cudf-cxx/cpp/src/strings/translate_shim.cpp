#include "strings/translate_shim.h"
#include <cudf/strings/translate.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <vector>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_translate(
    const OwnedColumn& col,
    rust::Slice<const uint32_t> src_chars,
    rust::Slice<const uint32_t> dst_chars)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> table;
    auto n = std::min(src_chars.size(), dst_chars.size());
    table.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        table.emplace_back(
            static_cast<cudf::char_utf8>(src_chars[i]),
            static_cast<cudf::char_utf8>(dst_chars[i]));
    }

    auto result = cudf::strings::translate(col.view(), table, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_filter_characters(
    const OwnedColumn& col,
    rust::Slice<const uint32_t> range_pairs,
    bool keep,
    rust::Str replacement)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> ranges;
    for (size_t i = 0; i + 1 < range_pairs.size(); i += 2) {
        ranges.emplace_back(
            static_cast<cudf::char_utf8>(range_pairs[i]),
            static_cast<cudf::char_utf8>(range_pairs[i + 1]));
    }

    auto filter = keep ? cudf::strings::filter_type::KEEP
                       : cudf::strings::filter_type::REMOVE;
    std::string repl(replacement.data(), replacement.size());
    cudf::string_scalar scalar_repl(repl, true, stream);

    auto result = cudf::strings::filter_characters(
        col.view(), ranges, filter, scalar_repl, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
