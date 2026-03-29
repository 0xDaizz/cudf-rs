#include "lists/lists_shim.h"
#include <cudf/lists/explode.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/extract.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

// ── Explode ────────────────────────────────────────────────────

std::unique_ptr<OwnedTable> lists_explode(
    const OwnedTable& table,
    int32_t explode_col_idx)
{
    auto result = cudf::explode(
        table.view(),
        static_cast<cudf::size_type>(explode_col_idx));
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> lists_explode_outer(
    const OwnedTable& table,
    int32_t explode_col_idx)
{
    auto result = cudf::explode_outer(
        table.view(),
        static_cast<cudf::size_type>(explode_col_idx));
    return std::make_unique<OwnedTable>(std::move(result));
}

// ── Sorting ───────────────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_sort(
    const OwnedColumn& col,
    bool ascending,
    int32_t null_order)
{
    auto lists_view = cudf::lists_column_view(col.view());
    auto order = ascending ? cudf::order::ASCENDING : cudf::order::DESCENDING;
    auto null_prec = null_order == 0 ? cudf::null_order::BEFORE : cudf::null_order::AFTER;

    auto result = cudf::lists::sort_lists(
        lists_view,
        order,
        null_prec);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Contains ──────────────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_contains(
    const OwnedColumn& col,
    const OwnedScalar& search_key)
{
    auto lists_view = cudf::lists_column_view(col.view());
    auto result = cudf::lists::contains(
        lists_view,
        *search_key.inner);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> lists_contains_nulls(
    const OwnedColumn& col)
{
    auto lists_view = cudf::lists_column_view(col.view());
    auto result = cudf::lists::contains_nulls(lists_view);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Extract ───────────────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_extract(
    const OwnedColumn& col,
    int32_t index)
{
    auto lists_view = cudf::lists_column_view(col.view());
    auto result = cudf::lists::extract_list_element(
        lists_view,
        static_cast<cudf::size_type>(index));
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
