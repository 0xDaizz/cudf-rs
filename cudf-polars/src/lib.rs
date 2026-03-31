//! GPU execution engine for Polars using NVIDIA libcudf.

pub mod convert;

#[cfg(test)]
mod tests {
    use super::*;
    use polars_core::prelude::*;

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_i32() {
        let df = df!("x" => [1i32, 2, 3, 4, 5]).unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        let orig = df.column("x").unwrap().i32().unwrap();
        let result = back.column("x").unwrap().i32().unwrap();
        assert_eq!(orig.len(), result.len());
        for i in 0..orig.len() {
            assert_eq!(orig.get(i), result.get(i), "mismatch at index {}", i);
        }
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_f64() {
        let df = df!("val" => [1.1f64, 2.2, 3.3]).unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        let orig = df.column("val").unwrap().f64().unwrap();
        let result = back.column("val").unwrap().f64().unwrap();
        assert_eq!(orig.len(), result.len());
        for i in 0..orig.len() {
            let o = orig.get(i).unwrap();
            let r = result.get(i).unwrap();
            assert!((o - r).abs() < f64::EPSILON, "mismatch at index {}: {} vs {}", i, o, r);
        }
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_string() {
        let df = df!("s" => ["hello", "world", "gpu"]).unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        let orig = df.column("s").unwrap().str().unwrap();
        let result = back.column("s").unwrap().str().unwrap();
        assert_eq!(orig.len(), result.len());
        for i in 0..orig.len() {
            assert_eq!(orig.get(i), result.get(i), "mismatch at index {}", i);
        }
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_multi_column() {
        let df = df!(
            "id" => [1i64, 2, 3],
            "value" => [10.0f64, 20.0, 30.0],
            "name" => ["a", "b", "c"]
        )
        .unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        assert_eq!(df.height(), back.height());
        assert_eq!(df.width(), back.width());
        // Check id column
        let orig_id = df.column("id").unwrap().i64().unwrap();
        let result_id = back.column("id").unwrap().i64().unwrap();
        for i in 0..orig_id.len() {
            assert_eq!(orig_id.get(i), result_id.get(i), "id mismatch at index {}", i);
        }
        // Check value column
        let orig_val = df.column("value").unwrap().f64().unwrap();
        let result_val = back.column("value").unwrap().f64().unwrap();
        for i in 0..orig_val.len() {
            let o = orig_val.get(i).unwrap();
            let r = result_val.get(i).unwrap();
            assert!((o - r).abs() < f64::EPSILON, "value mismatch at index {}: {} vs {}", i, o, r);
        }
        // Check name column
        let orig_name = df.column("name").unwrap().str().unwrap();
        let result_name = back.column("name").unwrap().str().unwrap();
        for i in 0..orig_name.len() {
            assert_eq!(orig_name.get(i), result_name.get(i), "name mismatch at index {}", i);
        }
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_boolean() {
        let df = df!("flag" => [true, false, true]).unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        let orig = df.column("flag").unwrap().bool().unwrap();
        let result = back.column("flag").unwrap().bool().unwrap();
        assert_eq!(orig.len(), result.len());
        for i in 0..orig.len() {
            assert_eq!(orig.get(i), result.get(i), "mismatch at index {}", i);
        }
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_nullable_i32() {
        let df = df!("x" => &[Some(1i32), None, Some(3), None, Some(5)]).unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        let orig = df.column("x").unwrap().i32().unwrap();
        let result = back.column("x").unwrap().i32().unwrap();
        assert_eq!(orig.len(), result.len());
        for i in 0..orig.len() {
            assert_eq!(orig.get(i), result.get(i), "mismatch at index {}", i);
        }
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_empty() {
        let df = df!("x" => Vec::<i32>::new()).unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        assert_eq!(back.height(), 0);
        assert_eq!(back.width(), 1);
    }
}
