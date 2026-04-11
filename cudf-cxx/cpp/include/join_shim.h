#pragma once

#include <cudf/join/join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/mark_join.hpp>
#include <memory>
#include "rust/cxx.h"
#include "table_shim.h"

namespace cudf_shims {

// Inner/left/full return a 2-column table of [left_indices, right_indices].
std::unique_ptr<OwnedTable> inner_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys);
std::unique_ptr<OwnedTable> left_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys);
std::unique_ptr<OwnedTable> full_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys);

// Cross join returns the full cartesian product table.
std::unique_ptr<OwnedTable> cross_join(
    const OwnedTable& left, const OwnedTable& right);

// Semi/anti joins return a single-column table of left indices.
std::unique_ptr<OwnedTable> left_semi_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys);
std::unique_ptr<OwnedTable> left_anti_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys);

// ── Mark Join ──────────────────────────────────────────────────
// Returns a single-column table of gather-map indices for the build/left table.
// In libcudf v26.04.00, cudf::mark_join builds from the left table and probes the
// right table, returning left-table row indices.

std::unique_ptr<OwnedTable> mark_semi_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys);
std::unique_ptr<OwnedTable> mark_anti_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys);

// ── Hash Join ─────────────────────────────────────────────────

/// Opaque wrapper around cudf::hash_join.
/// The build table is pre-hashed at construction time for efficient
/// repeated probing.
struct OwnedHashJoin {
    std::unique_ptr<cudf::hash_join> inner;

    explicit OwnedHashJoin(std::unique_ptr<cudf::hash_join> hj)
        : inner(std::move(hj)) {}
};

/// Create a hash join object from the build (right) table keys.
std::unique_ptr<OwnedHashJoin> hash_join_create(const OwnedTable& build);

/// Probe the hash join with an inner join, returning gather maps.
std::unique_ptr<OwnedTable> hash_join_inner(
    const OwnedHashJoin& hj, const OwnedTable& probe);

/// Probe the hash join with a left join, returning gather maps.
std::unique_ptr<OwnedTable> hash_join_left(
    const OwnedHashJoin& hj, const OwnedTable& probe);

/// Probe the hash join with a full outer join, returning gather maps.
std::unique_ptr<OwnedTable> hash_join_full(
    const OwnedHashJoin& hj, const OwnedTable& probe);

/// Get the estimated output size for inner join.
int64_t hash_join_inner_size(const OwnedHashJoin& hj, const OwnedTable& probe);

/// Get the estimated output size for left join.
int64_t hash_join_left_size(const OwnedHashJoin& hj, const OwnedTable& probe);

/// Get the estimated output size for full join.
int64_t hash_join_full_size(const OwnedHashJoin& hj, const OwnedTable& probe);

} // namespace cudf_shims
