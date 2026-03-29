#include "dictionary_shim.h"
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/search.hpp>
#include <cudf/dictionary/update_keys.hpp>
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

std::unique_ptr<OwnedScalar> dictionary_get_index(
    const OwnedColumn& col,
    const OwnedScalar& key)
{
    auto dict_view = cudf::dictionary_column_view(col.view());
    auto result = cudf::dictionary::get_index(dict_view, *key.inner);
    return std::make_unique<OwnedScalar>(std::move(result));
}

std::unique_ptr<OwnedColumn> dictionary_add_keys(
    const OwnedColumn& col,
    const OwnedColumn& new_keys)
{
    auto dict_view = cudf::dictionary_column_view(col.view());
    auto result = cudf::dictionary::add_keys(dict_view, new_keys.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> dictionary_remove_keys(
    const OwnedColumn& col,
    const OwnedColumn& keys_to_remove)
{
    auto dict_view = cudf::dictionary_column_view(col.view());
    auto result = cudf::dictionary::remove_keys(dict_view, keys_to_remove.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> dictionary_remove_unused_keys(
    const OwnedColumn& col)
{
    auto dict_view = cudf::dictionary_column_view(col.view());
    auto result = cudf::dictionary::remove_unused_keys(dict_view);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> dictionary_set_keys(
    const OwnedColumn& col,
    const OwnedColumn& new_keys)
{
    auto dict_view = cudf::dictionary_column_view(col.view());
    auto result = cudf::dictionary::set_keys(dict_view, new_keys.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<DictionaryMatchBuilder> dictionary_match_builder_new() {
    return std::make_unique<DictionaryMatchBuilder>();
}

std::unique_ptr<OwnedTable> dictionary_match_dictionaries(
    std::unique_ptr<DictionaryMatchBuilder> builder)
{
    // Build vector of dictionary_column_view from the builder's columns.
    std::vector<cudf::dictionary_column_view> dict_views;
    dict_views.reserve(builder->columns.size());
    for (auto& col : builder->columns) {
        dict_views.emplace_back(col->view());
    }

    auto results = cudf::dictionary::match_dictionaries(dict_views);

    // Package into a table.
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(results.size());
    for (auto& r : results) {
        cols.push_back(std::move(r));
    }
    auto table = std::make_unique<cudf::table>(std::move(cols));
    return std::make_unique<OwnedTable>(std::move(table));
}

} // namespace cudf_shims
