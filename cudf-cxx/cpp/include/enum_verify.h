#pragma once
#include <cudf/types.hpp>
#include <cudf/sorting.hpp>
#include <cudf/io/types.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/unary.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/null_mask.hpp>

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
static_assert(static_cast<int>(cudf::binary_operator::PMOD) == 7);
static_assert(static_cast<int>(cudf::binary_operator::PYMOD) == 8);
static_assert(static_cast<int>(cudf::binary_operator::POW) == 9);
static_assert(static_cast<int>(cudf::binary_operator::INT_POW) == 10);
static_assert(static_cast<int>(cudf::binary_operator::LOG_BASE) == 11);
static_assert(static_cast<int>(cudf::binary_operator::ATAN2) == 12);
static_assert(static_cast<int>(cudf::binary_operator::SHIFT_LEFT) == 13);
static_assert(static_cast<int>(cudf::binary_operator::SHIFT_RIGHT) == 14);
static_assert(static_cast<int>(cudf::binary_operator::SHIFT_RIGHT_UNSIGNED) == 15);
static_assert(static_cast<int>(cudf::binary_operator::BITWISE_AND) == 16);
static_assert(static_cast<int>(cudf::binary_operator::BITWISE_OR) == 17);
static_assert(static_cast<int>(cudf::binary_operator::BITWISE_XOR) == 18);
static_assert(static_cast<int>(cudf::binary_operator::LOGICAL_AND) == 19);
static_assert(static_cast<int>(cudf::binary_operator::LOGICAL_OR) == 20);
static_assert(static_cast<int>(cudf::binary_operator::EQUAL) == 21);
static_assert(static_cast<int>(cudf::binary_operator::NOT_EQUAL) == 22);
static_assert(static_cast<int>(cudf::binary_operator::LESS) == 23);
static_assert(static_cast<int>(cudf::binary_operator::GREATER) == 24);
static_assert(static_cast<int>(cudf::binary_operator::LESS_EQUAL) == 25);
static_assert(static_cast<int>(cudf::binary_operator::GREATER_EQUAL) == 26);
static_assert(static_cast<int>(cudf::binary_operator::NULL_EQUALS) == 27);
static_assert(static_cast<int>(cudf::binary_operator::NULL_NOT_EQUALS) == 28);
static_assert(static_cast<int>(cudf::binary_operator::NULL_MAX) == 29);
static_assert(static_cast<int>(cudf::binary_operator::NULL_MIN) == 30);
static_assert(static_cast<int>(cudf::binary_operator::GENERIC_BINARY) == 31);
static_assert(static_cast<int>(cudf::binary_operator::NULL_LOGICAL_AND) == 32);
static_assert(static_cast<int>(cudf::binary_operator::NULL_LOGICAL_OR) == 33);
static_assert(static_cast<int>(cudf::binary_operator::INVALID_BINARY) == 34);

// ── UnaryOp sync verification ───────────────────────────────
static_assert(static_cast<int>(cudf::unary_operator::SIN) == 0);
static_assert(static_cast<int>(cudf::unary_operator::COS) == 1);
static_assert(static_cast<int>(cudf::unary_operator::TAN) == 2);
static_assert(static_cast<int>(cudf::unary_operator::ARCSIN) == 3);
static_assert(static_cast<int>(cudf::unary_operator::ARCCOS) == 4);
static_assert(static_cast<int>(cudf::unary_operator::ARCTAN) == 5);
static_assert(static_cast<int>(cudf::unary_operator::SINH) == 6);
static_assert(static_cast<int>(cudf::unary_operator::COSH) == 7);
static_assert(static_cast<int>(cudf::unary_operator::TANH) == 8);
static_assert(static_cast<int>(cudf::unary_operator::ARCSINH) == 9);
static_assert(static_cast<int>(cudf::unary_operator::ARCCOSH) == 10);
static_assert(static_cast<int>(cudf::unary_operator::ARCTANH) == 11);
static_assert(static_cast<int>(cudf::unary_operator::EXP) == 12);
static_assert(static_cast<int>(cudf::unary_operator::LOG) == 13);
static_assert(static_cast<int>(cudf::unary_operator::SQRT) == 14);
static_assert(static_cast<int>(cudf::unary_operator::CBRT) == 15);
static_assert(static_cast<int>(cudf::unary_operator::CEIL) == 16);
static_assert(static_cast<int>(cudf::unary_operator::FLOOR) == 17);
static_assert(static_cast<int>(cudf::unary_operator::ABS) == 18);
static_assert(static_cast<int>(cudf::unary_operator::RINT) == 19);
static_assert(static_cast<int>(cudf::unary_operator::BIT_COUNT) == 20);
static_assert(static_cast<int>(cudf::unary_operator::BIT_INVERT) == 21);
static_assert(static_cast<int>(cudf::unary_operator::NOT) == 22);
static_assert(static_cast<int>(cudf::unary_operator::NEGATE) == 23);

// ── Compression sync verification ────────────────────────────
static_assert(static_cast<int>(cudf::io::compression_type::NONE) == 0);
static_assert(static_cast<int>(cudf::io::compression_type::AUTO) == 1);
static_assert(static_cast<int>(cudf::io::compression_type::SNAPPY) == 2);
static_assert(static_cast<int>(cudf::io::compression_type::GZIP) == 3);
static_assert(static_cast<int>(cudf::io::compression_type::BZIP2) == 4);
static_assert(static_cast<int>(cudf::io::compression_type::BROTLI) == 5);
static_assert(static_cast<int>(cudf::io::compression_type::ZIP) == 6);
static_assert(static_cast<int>(cudf::io::compression_type::XZ) == 7);
static_assert(static_cast<int>(cudf::io::compression_type::ZLIB) == 8);
static_assert(static_cast<int>(cudf::io::compression_type::LZ4) == 9);
static_assert(static_cast<int>(cudf::io::compression_type::LZO) == 10);
static_assert(static_cast<int>(cudf::io::compression_type::ZSTD) == 11);

// ── Order enum verification ──────────────────────────────────
static_assert(static_cast<int>(cudf::order::ASCENDING) == 0);
static_assert(static_cast<int>(cudf::order::DESCENDING) == 1);
static_assert(static_cast<int>(cudf::null_order::AFTER) == 0);
static_assert(static_cast<int>(cudf::null_order::BEFORE) == 1);

// ── DuplicateKeepOption sync verification ───────────────────
static_assert(static_cast<int>(cudf::duplicate_keep_option::KEEP_ANY) == 0);
static_assert(static_cast<int>(cudf::duplicate_keep_option::KEEP_FIRST) == 1);
static_assert(static_cast<int>(cudf::duplicate_keep_option::KEEP_LAST) == 2);
static_assert(static_cast<int>(cudf::duplicate_keep_option::KEEP_NONE) == 3);

// ── RankMethod sync verification ────────────────────────────
static_assert(static_cast<int>(cudf::rank_method::FIRST) == 0);
static_assert(static_cast<int>(cudf::rank_method::AVERAGE) == 1);
static_assert(static_cast<int>(cudf::rank_method::MIN) == 2);
static_assert(static_cast<int>(cudf::rank_method::MAX) == 3);
static_assert(static_cast<int>(cudf::rank_method::DENSE) == 4);

// ── NullEquality sync verification ──────────────────────────
static_assert(static_cast<int>(cudf::null_equality::EQUAL) == 0);
static_assert(static_cast<int>(cudf::null_equality::UNEQUAL) == 1);

// ── NullHandling (null_policy) sync verification ────────────
static_assert(static_cast<int>(cudf::null_policy::EXCLUDE) == 0);
static_assert(static_cast<int>(cudf::null_policy::INCLUDE) == 1);

// ── Interpolation sync verification ─────────────────────────
static_assert(static_cast<int>(cudf::interpolation::LINEAR) == 0);
static_assert(static_cast<int>(cudf::interpolation::LOWER) == 1);
static_assert(static_cast<int>(cudf::interpolation::HIGHER) == 2);
static_assert(static_cast<int>(cudf::interpolation::MIDPOINT) == 3);
static_assert(static_cast<int>(cudf::interpolation::NEAREST) == 4);

// ── MaskState sync verification ─────────────────────────────
static_assert(static_cast<int>(cudf::mask_state::UNALLOCATED) == 0);
static_assert(static_cast<int>(cudf::mask_state::UNINITIALIZED) == 1);
static_assert(static_cast<int>(cudf::mask_state::ALL_VALID) == 2);
static_assert(static_cast<int>(cudf::mask_state::ALL_NULL) == 3);
