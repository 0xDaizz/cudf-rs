#include "dictionary_shim.h"
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> dictionary_encode(
    const OwnedColumn& col)
{
    auto result = cudf::dictionary::encode(col.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> dictionary_decode(
    const OwnedColumn& col)
{
    auto dict_view = cudf::dictionary_column_view(col.view());
    auto result = cudf::dictionary::decode(dict_view);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
