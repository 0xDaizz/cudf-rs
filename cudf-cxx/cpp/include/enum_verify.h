#pragma once
#include <cudf/types.hpp>
#include <cudf/sorting.hpp>
#include <cudf/io/types.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/unaryop.hpp>

// ── TypeId sync verification ─────────────────────────────────
static_assert(static_cast<int>(cudf::type_id::EMPTY) == 0);
static_assert(static_cast<int>(cudf::type_id::INT8) == 1);
static_assert(static_cast<int>(cudf::type_id::INT16) == 2);
static_assert(static_cast<int>(cudf::type_id::INT32) == 3);
static_assert(static_cast<int>(cudf::type_id::INT64) == 4);
static_assert(static_cast<int>(cudf::type_id::UINT8) == 5);
static_assert(static_cast<int>(cudf::type_id::UINT16) == 6);
static_assert(static_cast<int>(cudf::type_id::UINT32) == 7);
static_assert(static_cast<int>(cudf::type_id::UINT64) == 8);
static_assert(static_cast<int>(cudf::type_id::FLOAT32) == 9);
static_assert(static_cast<int>(cudf::type_id::FLOAT64) == 10);
static_assert(static_cast<int>(cudf::type_id::BOOL8) == 11);
static_assert(static_cast<int>(cudf::type_id::TIMESTAMP_DAYS) == 12);
static_assert(static_cast<int>(cudf::type_id::TIMESTAMP_SECONDS) == 13);
static_assert(static_cast<int>(cudf::type_id::TIMESTAMP_MILLISECONDS) == 14);
static_assert(static_cast<int>(cudf::type_id::TIMESTAMP_MICROSECONDS) == 15);
static_assert(static_cast<int>(cudf::type_id::TIMESTAMP_NANOSECONDS) == 16);
static_assert(static_cast<int>(cudf::type_id::DURATION_DAYS) == 17);
static_assert(static_cast<int>(cudf::type_id::DURATION_SECONDS) == 18);
static_assert(static_cast<int>(cudf::type_id::DURATION_MILLISECONDS) == 19);
static_assert(static_cast<int>(cudf::type_id::DURATION_MICROSECONDS) == 20);
static_assert(static_cast<int>(cudf::type_id::DURATION_NANOSECONDS) == 21);
static_assert(static_cast<int>(cudf::type_id::DICTIONARY32) == 22);
static_assert(static_cast<int>(cudf::type_id::STRING) == 23);
static_assert(static_cast<int>(cudf::type_id::LIST) == 24);
static_assert(static_cast<int>(cudf::type_id::DECIMAL32) == 25);
static_assert(static_cast<int>(cudf::type_id::DECIMAL64) == 26);
static_assert(static_cast<int>(cudf::type_id::DECIMAL128) == 27);
static_assert(static_cast<int>(cudf::type_id::STRUCT) == 28);

// ── BinaryOp sync verification ───────────────────────────────
static_assert(static_cast<int>(cudf::binary_operator::ADD) == 0);
static_assert(static_cast<int>(cudf::binary_operator::SUB) == 1);
static_assert(static_cast<int>(cudf::binary_operator::MUL) == 2);
static_assert(static_cast<int>(cudf::binary_operator::DIV) == 3);
static_assert(static_cast<int>(cudf::binary_operator::TRUE_DIV) == 4);
static_assert(static_cast<int>(cudf::binary_operator::FLOOR_DIV) == 5);
static_assert(static_cast<int>(cudf::binary_operator::MOD) == 6);

// ── Compression sync verification ────────────────────────────
static_assert(static_cast<int>(cudf::io::compression_type::NONE) == 0);
static_assert(static_cast<int>(cudf::io::compression_type::SNAPPY) == 2);
static_assert(static_cast<int>(cudf::io::compression_type::ZSTD) == 11);

// ── Order enum verification ──────────────────────────────────
static_assert(static_cast<int>(cudf::order::ASCENDING) == 0);
static_assert(static_cast<int>(cudf::order::DESCENDING) == 1);
static_assert(static_cast<int>(cudf::null_order::AFTER) == 0);
static_assert(static_cast<int>(cudf::null_order::BEFORE) == 1);
