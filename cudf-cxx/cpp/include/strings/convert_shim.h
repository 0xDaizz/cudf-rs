#pragma once

#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/convert/convert_ipv4.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_to_integers(
    const OwnedColumn& col, int32_t type_id);
std::unique_ptr<OwnedColumn> str_from_integers(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_to_floats(
    const OwnedColumn& col, int32_t type_id);
std::unique_ptr<OwnedColumn> str_from_floats(const OwnedColumn& col);

// Booleans
std::unique_ptr<OwnedColumn> str_to_booleans(
    const OwnedColumn& col, rust::Str true_str);
std::unique_ptr<OwnedColumn> str_from_booleans(
    const OwnedColumn& col, rust::Str true_str, rust::Str false_str);

// Timestamps
std::unique_ptr<OwnedColumn> str_to_timestamps(
    const OwnedColumn& col, rust::Str format, int32_t type_id);
std::unique_ptr<OwnedColumn> str_from_timestamps(
    const OwnedColumn& col, rust::Str format);

// Durations
std::unique_ptr<OwnedColumn> str_to_durations(
    const OwnedColumn& col, rust::Str format, int32_t type_id);
std::unique_ptr<OwnedColumn> str_from_durations(
    const OwnedColumn& col, rust::Str format);

// Fixed point
std::unique_ptr<OwnedColumn> str_to_fixed_point(
    const OwnedColumn& col, int32_t type_id, int32_t scale);
std::unique_ptr<OwnedColumn> str_from_fixed_point(const OwnedColumn& col);

// Type checks
std::unique_ptr<OwnedColumn> str_is_integer(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_is_float(const OwnedColumn& col);

// Hex
std::unique_ptr<OwnedColumn> str_hex_to_integers(
    const OwnedColumn& col, int32_t type_id);
std::unique_ptr<OwnedColumn> str_integers_to_hex(const OwnedColumn& col);

// IPv4
std::unique_ptr<OwnedColumn> str_ipv4_to_integers(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_integers_to_ipv4(const OwnedColumn& col);

// URL encoding
std::unique_ptr<OwnedColumn> str_url_encode(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_url_decode(const OwnedColumn& col);

} // namespace cudf_shims
