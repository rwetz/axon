// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Axon                                ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! A tiny, dependency-free ONNX writer for the tabular MLP — burn has no
//! native ONNX export, so we hand-encode the protobuf (proto3 wire format)
//! for a `Gemm`/`Relu` graph. Standardization is baked in as leading
//! `Sub`/`Div` nodes so the exported model takes the schema's raw encoded
//! row and returns class logits — or, for regression, trailing `Mul`/`Add`
//! nodes return the value in the target's own units. Verified against
//! onnxruntime (see PLAN.md M5).
//!
//! Only what ONNX needs is implemented: varint + length-delimited fields and
//! the handful of message types in the graph. CNN export is a follow-up.

use std::io::Write;
use std::path::Path;

// ── protobuf wire primitives (proto3) ─────────────────────────────────

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

/// A varint field (wire type 0, so the tag is just `field << 3`).
fn f_varint(out: &mut Vec<u8>, field: u32, val: u64) {
    put_varint(out, (field as u64) << 3);
    put_varint(out, val);
}

/// A length-delimited field (wire type 2): bytes, strings, sub-messages.
fn f_bytes(out: &mut Vec<u8>, field: u32, data: &[u8]) {
    put_varint(out, ((field as u64) << 3) | 2);
    put_varint(out, data.len() as u64);
    out.extend_from_slice(data);
}

fn f_str(out: &mut Vec<u8>, field: u32, s: &str) {
    f_bytes(out, field, s.as_bytes());
}

// ── ONNX message builders ─────────────────────────────────────────────

/// A trained dense layer: row-major `weight` of shape `[w_rows, w_cols]`.
/// `trans_b` says it is stored `[out, in]` (Gemm `transB`). It is explicit
/// because a square layer can't be told apart by shape.
pub struct Dense {
    pub weight: Vec<f32>,
    pub w_rows: usize,
    pub w_cols: usize,
    pub trans_b: bool,
    pub bias: Vec<f32>,
    pub out_dim: usize,
}

/// TensorProto initializer (float): dims (1), data_type=FLOAT (2),
/// name (8), raw_data little-endian (9).
fn tensor_proto(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
    let mut m = Vec::new();
    for &d in dims {
        f_varint(&mut m, 1, d as u64);
    }
    f_varint(&mut m, 2, 1); // FLOAT
    f_str(&mut m, 8, name);
    let mut raw = Vec::with_capacity(data.len() * 4);
    for &v in data {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    f_bytes(&mut m, 9, &raw);
    m
}

/// AttributeProto for a single INT (name (1), i (3), type=INT=2 (20)).
fn attr_int(name: &str, val: i64) -> Vec<u8> {
    let mut m = Vec::new();
    f_str(&mut m, 1, name);
    f_varint(&mut m, 3, val as u64);
    f_varint(&mut m, 20, 2);
    m
}

/// NodeProto: input (1), output (2), name (3), op_type (4), attribute (5).
fn node(
    op_type: &str,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: &[Vec<u8>],
) -> Vec<u8> {
    let mut m = Vec::new();
    for i in inputs {
        f_str(&mut m, 1, i);
    }
    for o in outputs {
        f_str(&mut m, 2, o);
    }
    f_str(&mut m, 3, name);
    f_str(&mut m, 4, op_type);
    for a in attrs {
        f_bytes(&mut m, 5, a);
    }
    m
}

/// A `[batch, features]` float ValueInfoProto (the symbolic batch dim keeps
/// the model batch-agnostic).
fn value_info_2d(name: &str, features: i64) -> Vec<u8> {
    // Dimension: dim_value (1) or dim_param (2).
    let mut batch_dim = Vec::new();
    f_str(&mut batch_dim, 2, "batch");
    let mut feat_dim = Vec::new();
    f_varint(&mut feat_dim, 1, features as u64);
    // TensorShapeProto: dim (1, repeated).
    let mut shape = Vec::new();
    f_bytes(&mut shape, 1, &batch_dim);
    f_bytes(&mut shape, 1, &feat_dim);
    // TypeProto.Tensor: elem_type=FLOAT (1), shape (2).
    let mut tensor_ty = Vec::new();
    f_varint(&mut tensor_ty, 1, 1);
    f_bytes(&mut tensor_ty, 2, &shape);
    // TypeProto: tensor_type (1).
    let mut ty = Vec::new();
    f_bytes(&mut ty, 1, &tensor_ty);
    // ValueInfoProto: name (1), type (2).
    let mut vi = Vec::new();
    f_str(&mut vi, 1, name);
    f_bytes(&mut vi, 2, &ty);
    vi
}

/// Build a complete ONNX ModelProto for `raw input → standardize →
/// (Gemm[, Relu])* → logits` (or `→ · std + mean` for regression, when
/// `rescale` is `Some((mean, std))`) and return its bytes.
fn build_model(
    features: usize,
    mean: &[f32],
    std: &[f32],
    layers: &[Dense],
    rescale: Option<(f32, f32)>,
) -> Vec<u8> {
    let outputs = layers.last().map_or(0, |l| l.out_dim);
    let mut nodes: Vec<Vec<u8>> = Vec::new();
    let mut inits: Vec<Vec<u8>> = Vec::new();

    // Standardization: (input - mean) / std.
    inits.push(tensor_proto("mean", &[features as i64], mean));
    inits.push(tensor_proto("std", &[features as i64], std));
    nodes.push(node(
        "Sub",
        "standardize_sub",
        &["input", "mean"],
        &["x_centered"],
        &[],
    ));
    nodes.push(node(
        "Div",
        "standardize_div",
        &["x_centered", "std"],
        &["x_std"],
        &[],
    ));

    // Dense stack: Gemm then Relu (no Relu after the last layer).
    let mut prev = String::from("x_std");
    for (i, layer) in layers.iter().enumerate() {
        let last = i + 1 == layers.len();
        let w_name = format!("W{i}");
        let b_name = format!("B{i}");
        inits.push(tensor_proto(
            &w_name,
            &[layer.w_rows as i64, layer.w_cols as i64],
            &layer.weight,
        ));
        inits.push(tensor_proto(&b_name, &[layer.out_dim as i64], &layer.bias));
        let trans_b = i64::from(layer.trans_b);
        let gemm_out = if last && rescale.is_none() {
            "output".to_string()
        } else {
            format!("g{i}")
        };
        nodes.push(node(
            "Gemm",
            &format!("gemm{i}"),
            &[&prev, &w_name, &b_name],
            &[&gemm_out],
            &[attr_int("transB", trans_b)],
        ));
        if !last {
            let relu_out = format!("h{i}");
            nodes.push(node(
                "Relu",
                &format!("relu{i}"),
                &[&gemm_out],
                &[&relu_out],
                &[],
            ));
            prev = relu_out;
        } else {
            prev = gemm_out;
        }
    }

    // Regression: standardized prediction → the target's own units.
    if let Some((y_mean, y_std)) = rescale {
        inits.push(tensor_proto("y_std", &[1], &[y_std]));
        inits.push(tensor_proto("y_mean", &[1], &[y_mean]));
        nodes.push(node(
            "Mul",
            "unscale_mul",
            &[&prev, "y_std"],
            &["y_scaled"],
            &[],
        ));
        nodes.push(node(
            "Add",
            "unscale_add",
            &["y_scaled", "y_mean"],
            &["output"],
            &[],
        ));
    }

    // GraphProto: node (1), name (2), initializer (5), input (11), output (12).
    let mut graph = Vec::new();
    for n in &nodes {
        f_bytes(&mut graph, 1, n);
    }
    f_str(&mut graph, 2, "axon_mlp");
    for init in &inits {
        f_bytes(&mut graph, 5, init);
    }
    f_bytes(&mut graph, 11, &value_info_2d("input", features as i64));
    f_bytes(&mut graph, 12, &value_info_2d("output", outputs as i64));

    // OperatorSetIdProto: version (2) — default ai.onnx domain.
    let mut opset = Vec::new();
    f_varint(&mut opset, 2, 13);

    // ModelProto: ir_version (1), producer_name (2), opset_import (8), graph (7).
    let mut model = Vec::new();
    f_varint(&mut model, 1, 7); // IR version 7 (ONNX 1.8)
    f_str(&mut model, 2, "axon");
    f_bytes(&mut model, 8, &opset);
    f_bytes(&mut model, 7, &graph);
    model
}

/// Write an MLP to `path` as ONNX. The graph takes the encoded row (named
/// "input") and returns logits or the regression value (named "output");
/// standardization is baked in. Atomic write (tmp + rename), like the run
/// store.
pub fn write_mlp(
    path: &Path,
    features: usize,
    mean: &[f32],
    std: &[f32],
    layers: &[Dense],
    rescale: Option<(f32, f32)>,
) -> std::io::Result<()> {
    let bytes = build_model(features, mean, std, layers, rescale);
    let tmp = path.with_extension("onnx.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.flush()?;
    }
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_matches_protobuf_spec() {
        let enc = |v| {
            let mut out = Vec::new();
            put_varint(&mut out, v);
            out
        };
        assert_eq!(enc(0), [0]);
        assert_eq!(enc(127), [127]);
        assert_eq!(enc(128), [0x80, 0x01]);
        assert_eq!(enc(300), [0xAC, 0x02]);
    }

    #[test]
    fn build_model_emits_nonempty_proto() {
        // 2->3->2 MLP; just ensure assembly runs and produces bytes.
        let layers = [Dense {
            weight: vec![0.0; 2 * 2],
            w_rows: 2,
            w_cols: 2,
            trans_b: false,
            bias: vec![0.0; 2],
            out_dim: 2,
        }];
        let bytes = build_model(2, &[0.0, 0.0], &[1.0, 1.0], &layers, None);
        assert!(bytes.len() > 32);
        let reg = build_model(2, &[0.0, 0.0], &[1.0, 1.0], &layers, Some((5.0, 2.0)));
        assert!(reg.len() > bytes.len());
    }
}
