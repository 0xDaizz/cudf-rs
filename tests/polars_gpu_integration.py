#!/usr/bin/env python3
"""
Polars GPU engine (cudf-polars) end-to-end integration tests.

Tests polars LazyFrame.collect(engine="gpu") against CPU results.
Self-contained — requires only `polars` (with cudf-polars backend installed).

Exit code 0 = all pass, 1 = any failure.
"""

import polars as pl
import sys
import math

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def run_test(name, lf):
    """Execute LazyFrame on both CPU and GPU, compare results exactly."""
    global passed, failed
    try:
        cpu = lf.collect()
        gpu = lf.collect(engine="gpu")
        if cpu.equals(gpu):
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            print(f"    CPU: {cpu}")
            print(f"    GPU: {gpu}")
            failed += 1
    except Exception as e:
        print(f"  FAIL: {name} — {e}")
        failed += 1


def run_test_approx(name, lf, rtol=1e-5):
    """For float results, use approximate comparison."""
    global passed, failed
    try:
        cpu = lf.collect()
        gpu = lf.collect(engine="gpu")
        if cpu.shape != gpu.shape:
            print(f"  FAIL: {name} — shape mismatch {cpu.shape} vs {gpu.shape}")
            failed += 1
            return
        all_close = True
        for col in cpu.columns:
            if cpu[col].dtype.is_float():
                diff = (cpu[col] - gpu[col]).abs().max()
                if diff is None or diff > rtol:
                    all_close = False
            else:
                if not cpu[col].equals(gpu[col]):
                    all_close = False
        if all_close:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            print(f"    CPU: {cpu}")
            print(f"    GPU: {gpu}")
            failed += 1
    except Exception as e:
        print(f"  FAIL: {name} — {e}")
        failed += 1


def run_test_error(name, lf):
    """Expect the GPU collect to raise an exception."""
    global passed, failed
    try:
        lf.collect(engine="gpu")
        print(f"  FAIL: {name} — expected error but succeeded")
        failed += 1
    except Exception:
        print(f"  PASS: {name} (raised as expected)")
        passed += 1


# ===================================================================
# 1. Roundtrip — various dtypes
# ===================================================================

def test_roundtrip():
    print("\n[1] Roundtrip (dtype preservation)")

    # Integer types
    df = pl.DataFrame({
        "i32": pl.Series([1, 2, 3], dtype=pl.Int32),
        "i64": pl.Series([10, 20, 30], dtype=pl.Int64),
    })
    run_test("roundtrip_integers", df.lazy())

    # Float types
    df = pl.DataFrame({
        "f32": pl.Series([1.5, 2.5, 3.5], dtype=pl.Float32),
        "f64": pl.Series([1.1, 2.2, 3.3], dtype=pl.Float64),
    })
    run_test("roundtrip_floats", df.lazy())

    # String
    df = pl.DataFrame({"s": ["alpha", "beta", "gamma"]})
    run_test("roundtrip_string", df.lazy())

    # Boolean
    df = pl.DataFrame({"b": [True, False, True]})
    run_test("roundtrip_bool", df.lazy())

    # Mixed dtypes in one frame
    df = pl.DataFrame({
        "i": [1, 2, 3],
        "f": [1.0, 2.0, 3.0],
        "s": ["a", "b", "c"],
        "b": [True, False, True],
    })
    run_test("roundtrip_mixed", df.lazy())

    # Empty dataframe
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Int64)})
    run_test("roundtrip_empty", df.lazy())


# ===================================================================
# 2. Filter
# ===================================================================

def test_filter():
    print("\n[2] Filter")

    df = pl.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50],
    }).lazy()

    run_test("filter_gt", df.filter(pl.col("a") > 3))
    run_test("filter_eq", df.filter(pl.col("a") == 2))
    run_test("filter_chained", df.filter((pl.col("a") > 1) & (pl.col("b") < 50)))
    run_test("filter_or", df.filter((pl.col("a") == 1) | (pl.col("a") == 5)))

    # Filter with nulls
    df_null = pl.DataFrame({
        "a": [1, None, 3, None, 5],
        "b": [10, 20, 30, 40, 50],
    }).lazy()
    run_test("filter_with_nulls", df_null.filter(pl.col("a") > 2))
    run_test("filter_is_not_null", df_null.filter(pl.col("a").is_not_null()))
    run_test("filter_is_null", df_null.filter(pl.col("a").is_null()))


# ===================================================================
# 3. Select
# ===================================================================

def test_select():
    print("\n[3] Select")

    df = pl.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9],
    }).lazy()

    run_test("select_subset", df.select("a", "c"))
    run_test("select_alias", df.select(pl.col("a").alias("x")))
    run_test("select_expr", df.select(pl.col("a") + pl.col("b")))
    run_test("select_literal", df.select(pl.lit(42).alias("const")))
    run_test("select_multiple_expr", df.select(
        pl.col("a") * 2,
        (pl.col("b") + pl.col("c")).alias("bc_sum"),
    ))


# ===================================================================
# 4. GroupBy
# ===================================================================

def test_groupby():
    print("\n[4] GroupBy")

    df = pl.DataFrame({
        "key": ["a", "a", "b", "b", "b"],
        "val": [1, 2, 3, 4, 5],
    }).lazy()

    # Sort after group_by to get deterministic order
    run_test("groupby_sum", df.group_by("key").agg(pl.col("val").sum()).sort("key"))
    run_test("groupby_min", df.group_by("key").agg(pl.col("val").min()).sort("key"))
    run_test("groupby_max", df.group_by("key").agg(pl.col("val").max()).sort("key"))
    run_test("groupby_count", df.group_by("key").agg(pl.col("val").count()).sort("key"))
    run_test_approx("groupby_mean", df.group_by("key").agg(pl.col("val").mean()).sort("key"))

    # Multiple aggregations
    run_test_approx("groupby_multi_agg", df.group_by("key").agg(
        pl.col("val").sum().alias("total"),
        pl.col("val").min().alias("low"),
        pl.col("val").max().alias("high"),
        pl.col("val").mean().alias("avg"),
    ).sort("key"))

    # Integer key
    df2 = pl.DataFrame({
        "grp": [1, 1, 2, 2, 3],
        "v": [10, 20, 30, 40, 50],
    }).lazy()
    run_test("groupby_int_key", df2.group_by("grp").agg(pl.col("v").sum()).sort("grp"))

    # Multiple keys
    df3 = pl.DataFrame({
        "k1": ["a", "a", "b", "b"],
        "k2": [1, 2, 1, 2],
        "v": [10, 20, 30, 40],
    }).lazy()
    run_test("groupby_multi_key", df3.group_by("k1", "k2").agg(pl.col("v").sum()).sort("k1", "k2"))


# ===================================================================
# 5. Sort
# ===================================================================

def test_sort():
    print("\n[5] Sort")

    df = pl.DataFrame({
        "a": [3, 1, 4, 1, 5],
        "b": [10, 20, 30, 40, 50],
    }).lazy()

    run_test("sort_asc", df.sort("a"))
    run_test("sort_desc", df.sort("a", descending=True))
    run_test("sort_multi", df.sort("a", "b"))
    run_test("sort_multi_mixed", df.sort("a", "b", descending=[True, False]))

    # Sort strings
    df_str = pl.DataFrame({"s": ["banana", "apple", "cherry"]}).lazy()
    run_test("sort_string", df_str.sort("s"))


# ===================================================================
# 6. Slice (limit / head / tail)
# ===================================================================

def test_slice():
    print("\n[6] Slice")

    df = pl.DataFrame({"a": list(range(20))}).lazy()

    run_test("slice_head", df.head(5))
    run_test("slice_tail", df.tail(5))
    run_test("slice_limit", df.limit(3))
    run_test("slice_offset_len", df.slice(5, 3))


# ===================================================================
# 7. Binary ops
# ===================================================================

def test_binary_ops():
    print("\n[7] Binary ops")

    df = pl.DataFrame({
        "a": [10, 20, 30],
        "b": [3, 4, 5],
    }).lazy()

    run_test("binop_add", df.select((pl.col("a") + pl.col("b")).alias("r")))
    run_test("binop_sub", df.select((pl.col("a") - pl.col("b")).alias("r")))
    run_test("binop_mul", df.select((pl.col("a") * pl.col("b")).alias("r")))
    run_test_approx("binop_div", df.select((pl.col("a") / pl.col("b")).alias("r")))
    run_test("binop_mod", df.select((pl.col("a") % pl.col("b")).alias("r")))

    # Comparison operators — produce boolean columns
    run_test("binop_gt", df.select((pl.col("a") > pl.col("b")).alias("r")))
    run_test("binop_lt", df.select((pl.col("a") < 15).alias("r")))
    run_test("binop_eq", df.select((pl.col("a") == 20).alias("r")))

    # Scalar arithmetic
    run_test("binop_scalar_mul", df.select((pl.col("a") * 2).alias("r")))
    run_test_approx("binop_scalar_div", df.select((pl.col("a") / 3).alias("r")))

    # Mixed int/float
    df_mix = pl.DataFrame({
        "i": pl.Series([1, 2, 3], dtype=pl.Int64),
        "f": pl.Series([0.5, 1.5, 2.5], dtype=pl.Float64),
    }).lazy()
    run_test_approx("binop_mixed_types", df_mix.select((pl.col("i") + pl.col("f")).alias("r")))


# ===================================================================
# 8. Join
# ===================================================================

def test_join():
    print("\n[8] Join")

    left = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "val_l": ["a", "b", "c", "d"],
    }).lazy()
    right = pl.DataFrame({
        "id": [2, 3, 4, 5],
        "val_r": [20, 30, 40, 50],
    }).lazy()

    run_test("join_inner", left.join(right, on="id", how="inner").sort("id"))
    run_test("join_left", left.join(right, on="id", how="left").sort("id"))

    # Join with different key names
    right2 = pl.DataFrame({
        "rid": [1, 2, 3],
        "val_r": [100, 200, 300],
    }).lazy()
    run_test("join_diff_keys", left.join(
        right2, left_on="id", right_on="rid", how="inner"
    ).sort("id"))

    # Self-join
    df = pl.DataFrame({"k": [1, 2, 3], "v": [10, 20, 30]}).lazy()
    run_test("join_self", df.join(df, on="k", how="inner").sort("k"))


# ===================================================================
# 9. Unique / Distinct
# ===================================================================

def test_unique():
    print("\n[9] Unique / Distinct")

    df = pl.DataFrame({"a": [1, 2, 2, 3, 3, 3]}).lazy()
    run_test("unique_single", df.unique().sort("a"))

    df2 = pl.DataFrame({
        "a": [1, 1, 2, 2],
        "b": ["x", "x", "y", "z"],
    }).lazy()
    run_test("unique_multi", df2.unique().sort("a", "b"))

    run_test("unique_subset", df2.unique(subset=["a"]).sort("a"))


# ===================================================================
# 10. with_columns
# ===================================================================

def test_with_columns():
    print("\n[10] with_columns")

    df = pl.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    }).lazy()

    run_test("with_columns_computed", df.with_columns(
        (pl.col("a") + pl.col("b")).alias("c"),
    ))
    run_test("with_columns_literal", df.with_columns(
        pl.lit("hello").alias("greeting"),
    ))
    run_test("with_columns_overwrite", df.with_columns(
        (pl.col("a") * 10).alias("a"),
    ))
    run_test("with_columns_multiple", df.with_columns(
        (pl.col("a") * 2).alias("a2"),
        (pl.col("b") - 1).alias("b1"),
        pl.lit(True).alias("flag"),
    ))


# ===================================================================
# 11. Concat
# ===================================================================

def test_concat():
    print("\n[11] Concat")

    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]}).lazy()
    df2 = pl.DataFrame({"a": [5, 6], "b": [7, 8]}).lazy()

    run_test("concat_vertical", pl.concat([df1, df2]))

    # Three frames
    df3 = pl.DataFrame({"a": [9], "b": [10]}).lazy()
    run_test("concat_three", pl.concat([df1, df2, df3]))

    # Concat with different row counts
    df_a = pl.DataFrame({"x": [1]}).lazy()
    df_b = pl.DataFrame({"x": list(range(100))}).lazy()
    run_test("concat_uneven", pl.concat([df_a, df_b]))


# ===================================================================
# 12. Large data
# ===================================================================

def test_large():
    print("\n[12] Large data (100k rows)")

    n = 100_000
    df = pl.DataFrame({
        "grp": [i % 100 for i in range(n)],
        "val": [float(i) for i in range(n)],
    }).lazy()

    run_test_approx("large_filter_agg",
        df.filter(pl.col("grp") < 50)
          .group_by("grp")
          .agg(pl.col("val").sum())
          .sort("grp")
    )

    run_test("large_sort", df.sort("val", descending=True).head(10))

    run_test_approx("large_with_columns",
        df.with_columns((pl.col("val") * 2).alias("doubled"))
          .filter(pl.col("grp") == 0)
          .select("val", "doubled")
          .sort("val")
    )


# ===================================================================
# 13. Nulls
# ===================================================================

def test_nulls():
    print("\n[13] Nulls")

    df = pl.DataFrame({
        "a": [1, None, 3, None, 5],
        "b": [None, 2, None, 4, None],
    }).lazy()

    run_test("nulls_roundtrip", df)
    run_test("nulls_fill", df.with_columns(pl.col("a").fill_null(0)))
    run_test("nulls_drop", df.drop_nulls())
    run_test("nulls_drop_subset", df.drop_nulls(subset=["a"]))

    # Aggregations with nulls
    run_test("nulls_sum", df.select(pl.col("a").sum()))
    run_test("nulls_count", df.select(pl.col("a").count()))
    run_test_approx("nulls_mean", df.select(pl.col("a").mean()))

    # Null in string column
    df_str = pl.DataFrame({"s": ["hello", None, "world"]}).lazy()
    run_test("nulls_string", df_str)
    run_test("nulls_string_filter", df_str.filter(pl.col("s").is_not_null()))

    # Null arithmetic (should propagate)
    run_test("nulls_arithmetic", df.select((pl.col("a") + pl.col("b")).alias("c")))


# ===================================================================
# 14. Fallback paths (GPU falls back to CPU for unsupported ops)
# ===================================================================

def test_fallback_paths():
    print("\n[14] Fallback paths (GPU falls back to CPU)")

    # Window functions — GPU engine falls back to CPU
    df = pl.DataFrame({
        "a": [1, 1, 2, 2, 3],
        "b": [10, 20, 30, 40, 50],
    }).lazy()

    run_test("fallback_window_over",
        df.with_columns(pl.col("b").sum().over("a").alias("b_sum"))
    )

    # Melt / unpivot — GPU engine falls back to CPU
    run_test("fallback_unpivot",
        df.unpivot(on=["a", "b"])
    )


# ===================================================================
# 15. Chained operations (pipeline test)
# ===================================================================

def test_pipeline():
    print("\n[15] Chained pipeline")

    n = 10_000
    df = pl.DataFrame({
        "category": [f"cat_{i % 5}" for i in range(n)],
        "region": [f"r_{i % 3}" for i in range(n)],
        "amount": [float(i % 1000) for i in range(n)],
        "flag": [i % 2 == 0 for i in range(n)],
    }).lazy()

    # Complex pipeline: filter -> with_columns -> group_by -> sort -> head
    # Complex pipeline: filter -> with_columns -> group_by -> agg
    # Skip .sort().head() since tie-breaking order differs between CPU/GPU
    global passed, failed
    pipeline_lf = (
        df.filter(pl.col("flag"))
          .with_columns((pl.col("amount") * 1.1).alias("adjusted"))
          .group_by("category", "region")
          .agg(
              pl.col("adjusted").sum().alias("total"),
              pl.col("adjusted").mean().alias("avg"),
              pl.col("adjusted").count().alias("n"),
          )
    )
    try:
        cpu = pipeline_lf.collect().sort("category", "region")
        gpu_result = pipeline_lf.collect(engine="gpu").sort("category", "region")
        if cpu.shape != gpu_result.shape:
            print(f"  FAIL: pipeline_full — shape mismatch")
            failed += 1
        else:
            all_close = True
            for c in cpu.columns:
                if cpu[c].dtype.is_float():
                    diff = (cpu[c] - gpu_result[c]).abs().max()
                    if diff is None or diff > 1e-5:
                        all_close = False
                else:
                    if not cpu[c].equals(gpu_result[c]):
                        all_close = False
            if all_close:
                print(f"  PASS: pipeline_full")
                passed += 1
            else:
                print(f"  FAIL: pipeline_full")
                print(f"    CPU: {cpu}")
                print(f"    GPU: {gpu_result}")
                failed += 1
    except Exception as e:
        print(f"  FAIL: pipeline_full — {e}")
        failed += 1

    # Filter -> join -> aggregate
    left = pl.DataFrame({
        "id": list(range(1000)),
        "val": [float(i) for i in range(1000)],
    }).lazy()
    right = pl.DataFrame({
        "id": list(range(0, 1000, 2)),
        "label": [f"label_{i}" for i in range(500)],
    }).lazy()

    run_test_approx("pipeline_join_agg",
        left.join(right, on="id", how="inner")
            .filter(pl.col("val") > 100)
            .with_columns((pl.col("val") * 2).alias("doubled"))
            .select("id", "val", "doubled", "label")
            .sort("id")
            .head(20)
    )


# ===================================================================
# 16. Cast operations
# ===================================================================

def test_cast():
    print("\n[16] Cast")

    df = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    run_test("cast_i64_to_f64", df.select(pl.col("a").cast(pl.Float64)))
    run_test("cast_i64_to_i32", df.select(pl.col("a").cast(pl.Int32)))

    df_f = pl.DataFrame({"x": [1.1, 2.9, 3.5]}).lazy()
    run_test("cast_f64_to_i64", df_f.select(pl.col("x").cast(pl.Int64)))

    df_bool = pl.DataFrame({"b": [True, False, True]}).lazy()
    run_test("cast_bool_to_int", df_bool.select(pl.col("b").cast(pl.Int64)))


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("Polars GPU Engine (cudf-polars) Integration Tests")
    print("=" * 60)
    print(f"Polars version: {pl.__version__}")

    test_roundtrip()
    test_filter()
    test_select()
    test_groupby()
    test_sort()
    test_slice()
    test_binary_ops()
    test_join()
    test_unique()
    test_with_columns()
    test_concat()
    test_large()
    test_nulls()
    test_fallback_paths()
    test_pipeline()
    test_cast()

    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nSome tests FAILED.")
        sys.exit(1)
    else:
        print("\nAll tests PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
