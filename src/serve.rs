// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Axon                                ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! `serve --run <id>` — the inference loop behind Nexis's Playground and
//! chat tools: one `ready` event, then one `prediction`/`error` per request
//! line on stdin.
//!
//! Serving needs no burn at all: `train` saves the tabular MLP's dense
//! weights plus the input schema, and a single row's forward pass is plain
//! `Vec<f32>` math (encode → Gemm/ReLU stack → softmax, or → the value).
//! Requests take raw column values — numbers or category strings; a missing
//! or blank value is treated the way training treated a missing cell.
//!
//! Image (CNN) runs and runs trained by engines < 0.8 have no servable
//! checkpoint — those get a clear `error` event instead of a crash.

use std::io::{BufRead, Write};
use std::path::Path;

use serde_json::json;

use crate::checkpoint::{self, Model, Prediction};
use crate::protocol::PROTOCOL_VERSION;
use crate::tabular::{Column, Raw, Task};

fn emit(event: serde_json::Value) {
    if let Ok(line) = serde_json::to_string(&event) {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    }
}

fn emit_error(msg: &str) {
    emit(json!({ "ev": "error", "msg": msg }));
    eprintln!("error: {msg}");
}

/// The `ready` event's `meta`: what a request should supply. `features` is
/// the input column names (the field older Nexis builds read); `columns`
/// adds each one's kind, and the categories for a categorical column, so
/// the Playground can offer a dropdown instead of a number box.
fn ready_meta(model: &Model) -> serde_json::Value {
    let columns: Vec<serde_json::Value> = model
        .schema
        .columns
        .iter()
        .map(|c| match c {
            Column::Numeric { name, mean, .. } => {
                json!({ "name": name, "kind": "numeric", "mean": mean })
            }
            Column::Categorical { name, categories } => {
                let real: Vec<&String> = categories
                    .iter()
                    .filter(|k| *k != crate::tabular::MISSING)
                    .collect();
                json!({ "name": name, "kind": "categorical", "categories": real })
            }
        })
        .collect();
    json!({
        "features": model.schema.input_names(),
        "columns": columns,
        "classes": model.classes,
        "task": model.task.as_str(),
        "target": model.target,
    })
}

fn predict(model: &Model, input: &serde_json::Value) -> serde_json::Value {
    let output = match model.predict(|name| Raw::from_json(input.get(name))) {
        Prediction::Class(best, probs) => {
            let probs_map: serde_json::Map<String, serde_json::Value> = model
                .classes
                .iter()
                .zip(probs.iter())
                .map(|(c, &p)| (c.clone(), json!(p)))
                .collect();
            json!({ "label": model.classes[best], "probs": probs_map })
        }
        Prediction::Value(v) => json!({ "value": v }),
    };
    json!({ "ev": "prediction", "input": input, "output": output })
}

/// The serve command: emit `ready`, then answer one request per stdin line
/// until EOF (Nexis closing the pipe / killing the session).
pub fn serve(project: &Path, run_id: &str) -> i32 {
    let dir = checkpoint::run_dir(project, run_id);
    if !dir.is_dir() {
        emit_error(&format!("run not found: {run_id}"));
        return 1;
    }
    let model = match checkpoint::load(&dir) {
        Ok(m) => m,
        Err(msg) => {
            emit_error(&msg);
            return 1;
        }
    };

    emit(json!({
        "ev": "ready",
        "run": run_id,
        "template": "tabular",
        "device": "cpu",
        "meta": ready_meta(&model),
        "protocol": PROTOCOL_VERSION,
    }));
    let outputs = match model.task {
        Task::Classification => format!("{} classes", model.classes.len()),
        Task::Regression => "a number".to_string(),
    };
    eprintln!(
        "serving run {run_id} ({} columns → {outputs}); one JSON request per line",
        model.schema.columns.len(),
    );

    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    let mut line = String::new();
    loop {
        line.clear();
        match lock.read_line(&mut line) {
            Ok(0) | Err(_) => return 0, // EOF — session closed
            Ok(_) => {}
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            emit_error("malformed request (expected one JSON object per line)");
            continue;
        };
        let Some(input) = req.get("input") else {
            emit_error(r#"request has no "input" field"#);
            continue;
        };
        emit(predict(&model, input));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{DenseWeights, WeightsFile};
    use crate::tabular::Schema;

    fn model(task: Task, classes: Vec<String>, schema: Schema, w: Vec<Vec<f32>>) -> Model {
        let (in_dim, out_dim) = (w.len(), w[0].len());
        Model {
            task,
            target: Some("t".into()),
            classes,
            schema,
            y_mean: 0.0,
            y_std: 1.0,
            weights: WeightsFile {
                layers: vec![DenseWeights {
                    in_dim,
                    out_dim,
                    w,
                    b: vec![0.0; out_dim],
                }],
            },
            trees: None,
        }
    }

    #[test]
    fn predict_labels_the_argmax_class() {
        let m = model(
            Task::Classification,
            vec!["no".into(), "yes".into()],
            Schema::legacy(&["f".into()], &[0.0], &[1.0]),
            vec![vec![-1.0, 1.0]],
        );
        let out = predict(&m, &json!({ "f": 5.0 }));
        assert_eq!(out["ev"], "prediction");
        assert_eq!(out["output"]["label"], "yes");
        assert!(out["output"]["probs"]["yes"].as_f64().unwrap() > 0.99);
    }

    #[test]
    fn categorical_input_selects_its_one_hot_slot() {
        let schema = Schema {
            columns: vec![Column::Categorical {
                name: "color".into(),
                categories: vec!["(missing)".into(), "blue".into(), "red".into()],
            }],
        };
        // Only the "red" slot pushes the value up.
        let m = model(
            Task::Regression,
            vec![],
            schema,
            vec![vec![0.0], vec![0.0], vec![7.0]],
        );
        let out = predict(&m, &json!({ "color": "red" }));
        assert_eq!(out["output"]["value"], 7.0);
        let meta = ready_meta(&m);
        assert_eq!(meta["task"], "regression");
        // The missing marker is an encoding detail, not a choice to offer.
        assert_eq!(meta["columns"][0]["categories"], json!(["blue", "red"]));
    }
}
