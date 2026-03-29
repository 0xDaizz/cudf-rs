#pragma once

#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/search.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/types.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"
#include "scalar_shim.h"
#include "table_shim.h"

namespace cudf_shims {

/// Dictionary-encode a column.
std::unique_ptr<OwnedColumn> dictionary_encode(
    const OwnedColumn& col);

/// Decode a dictionary column back to its original representation.
std::unique_ptr<OwnedColumn> dictionary_decode(
    const OwnedColumn& col);

/// Get the index of a key in a dictionary column. Returns a scalar.
std::unique_ptr<OwnedScalar> dictionary_get_index(
    const OwnedColumn& col,
    const OwnedScalar& key);

/// Add new keys to a dictionary column.
std::unique_ptr<OwnedColumn> dictionary_add_keys(
    const OwnedColumn& col,
    const OwnedColumn& new_keys);

/// Remove specified keys from a dictionary column.
std::unique_ptr<OwnedColumn> dictionary_remove_keys(
    const OwnedColumn& col,
    const OwnedColumn& keys_to_remove);

/// Remove unused keys from a dictionary column.
std::unique_ptr<OwnedColumn> dictionary_remove_unused_keys(
    const OwnedColumn& col);

/// Replace all keys in a dictionary column with new keys.
std::unique_ptr<OwnedColumn> dictionary_set_keys(
    const OwnedColumn& col,
    const OwnedColumn& new_keys);

/// Builder for collecting dictionary columns to pass to match_dictionaries.
struct DictionaryMatchBuilder {
    std::vector<std::unique_ptr<OwnedColumn>> columns;

    void add_column(std::unique_ptr<OwnedColumn> col) {
        columns.push_back(std::move(col));
    }

    int32_t num_columns() const {
        return static_cast<int32_t>(columns.size());
    }
};

std::unique_ptr<DictionaryMatchBuilder> dictionary_match_builder_new();

/// Match dictionaries of multiple columns so they share the same key set.
/// Returns a table with the matched columns.
std::unique_ptr<OwnedTable> dictionary_match_dictionaries(
    std::unique_ptr<DictionaryMatchBuilder> builder);

} // namespace cudf_shims
