#include "lists/lists_shim.h"
#include <cudf/lists/explode.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/extract.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/lists/gather.hpp>
#include <cudf/lists/set_operations.hpp>
#include <cudf/lists/reverse.hpp>
#include <cudf/lists/stream_compaction.hpp>
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
    auto null_prec = null_order == 0 ? cudf::null_order::AFTER : cudf::null_order::BEFORE;

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

// ── Explode Position ──────────────────────────────────────────

std::unique_ptr<OwnedTable> lists_explode_position(
    const OwnedTable& table,
    int32_t explode_col_idx)
{
    auto result = cudf::explode_position(
        table.view(),
        static_cast<cudf::size_type>(explode_col_idx));
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> lists_explode_outer_position(
    const OwnedTable& table,
    int32_t explode_col_idx)
{
    auto result = cudf::explode_outer_position(
        table.view(),
        static_cast<cudf::size_type>(explode_col_idx));
    return std::make_unique<OwnedTable>(std::move(result));
}

// ── Count Elements ────────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_count_elements(
    const OwnedColumn& col)
{
    auto lists_view = cudf::lists_column_view(col.view());
    auto result = cudf::lists::count_elements(lists_view);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Index Of (scalar) ─────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_index_of_scalar(
    const OwnedColumn& col,
    const OwnedScalar& key)
{
    auto lists_view = cudf::lists_column_view(col.view());
    auto result = cudf::lists::index_of(
        lists_view,
        *key.inner,
        cudf::lists::duplicate_find_option::FIND_FIRST);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Combine ───────────────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_concatenate_rows(
    const OwnedTable& table)
{
    auto result = cudf::lists::concatenate_rows(table.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> lists_concatenate_list_elements(
    const OwnedColumn& col)
{
    auto result = cudf::lists::concatenate_list_elements(col.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Filling (sequences) ──────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_sequences(
    const OwnedColumn& starts,
    const OwnedColumn& sizes)
{
    auto result = cudf::lists::sequences(starts.view(), sizes.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Gather ────────────────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_segmented_gather(
    const OwnedColumn& col,
    const OwnedColumn& gather_map)
{
    auto lists_view = cudf::lists_column_view(col.view());
    auto map_view = cudf::lists_column_view(gather_map.view());
    auto result = cudf::lists::segmented_gather(lists_view, map_view);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Set Operations ────────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_have_overlap(
    const OwnedColumn& lhs,
    const OwnedColumn& rhs)
{
    auto lv = cudf::lists_column_view(lhs.view());
    auto rv = cudf::lists_column_view(rhs.view());
    auto result = cudf::lists::have_overlap(lv, rv);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> lists_intersect_distinct(
    const OwnedColumn& lhs,
    const OwnedColumn& rhs)
{
    auto lv = cudf::lists_column_view(lhs.view());
    auto rv = cudf::lists_column_view(rhs.view());
    auto result = cudf::lists::intersect_distinct(lv, rv);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> lists_union_distinct(
    const OwnedColumn& lhs,
    const OwnedColumn& rhs)
{
    auto lv = cudf::lists_column_view(lhs.view());
    auto rv = cudf::lists_column_view(rhs.view());
    auto result = cudf::lists::union_distinct(lv, rv);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> lists_difference_distinct(
    const OwnedColumn& lhs,
    const OwnedColumn& rhs)
{
    auto lv = cudf::lists_column_view(lhs.view());
    auto rv = cudf::lists_column_view(rhs.view());
    auto result = cudf::lists::difference_distinct(lv, rv);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Reverse ───────────────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_reverse(
    const OwnedColumn& col)
{
    auto lists_view = cudf::lists_column_view(col.view());
    auto result = cudf::lists::reverse(lists_view);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Stream Compaction ─────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_apply_boolean_mask(
    const OwnedColumn& col,
    const OwnedColumn& mask)
{
    auto lists_view = cudf::lists_column_view(col.view());
    auto mask_view = cudf::lists_column_view(mask.view());
    auto result = cudf::lists::apply_boolean_mask(lists_view, mask_view);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> lists_distinct(
    const OwnedColumn& col)
{
    auto lists_view = cudf::lists_column_view(col.view());
    auto result = cudf::lists::distinct(lists_view);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── New Low Priority ─────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_stable_sort(
    const OwnedColumn& col,
    bool ascending,
    int32_t null_order)
{
    auto lists_view = cudf::lists_column_view(col.view());
    auto order = ascending ? cudf::order::ASCENDING : cudf::order::DESCENDING;
    auto null_prec = null_order == 0 ? cudf::null_order::AFTER : cudf::null_order::BEFORE;

    auto result = cudf::lists::stable_sort_lists(
        lists_view, order, null_prec);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> lists_extract_column_index(
    const OwnedColumn& col,
    const OwnedColumn& indices)
{
    auto lists_view = cudf::lists_column_view(col.view());
    auto result = cudf::lists::extract_list_element(
        lists_view, indices.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> lists_contains_column(
    const OwnedColumn& col,
    const OwnedColumn& search_keys)
{
    auto lists_view = cudf::lists_column_view(col.view());
    auto result = cudf::lists::contains(
        lists_view, search_keys.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
