#pragma once

#include <cudf/concatenate.hpp>
#include <cudf/types.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

// ── Concatenation Builder ─────────────────────────────────────
//
// cxx cannot directly pass Vec<&OwnedColumn>. We use a builder
// pattern: Rust creates a builder, adds columns/tables one by one,
// then calls build() to perform the concatenation.

/// Builder for concatenating columns.
struct ColumnConcatBuilder {
    std::vector<cudf::column_view> views;

    /// Add a column to the concatenation.
    void add(const OwnedColumn& col) {
        views.push_back(col.view());
    }

    /// Perform the concatenation and return the result.
    std::unique_ptr<OwnedColumn> build() const;
};

/// Builder for concatenating tables.
struct TableConcatBuilder {
    std::vector<cudf::table_view> views;

    /// Add a table to the concatenation.
    void add(const OwnedTable& table) {
        views.push_back(table.view());
    }

    /// Perform the concatenation and return the result.
    std::unique_ptr<OwnedTable> build() const;
};

/// Create a new ColumnConcatBuilder.
std::unique_ptr<ColumnConcatBuilder> new_column_concat_builder();

/// Create a new TableConcatBuilder.
std::unique_ptr<TableConcatBuilder> new_table_concat_builder();

} // namespace cudf_shims
