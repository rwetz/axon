// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Axon                                ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! Tabular data: CSV reading, column typing, the task decision, and the
//! fitted [`Schema`] that turns a raw row into the model's input vector.
//!
//! The schema is the contract between `train`, `serve` and `export`. It is
//! fitted on the training split only, saved into the checkpoint, and reused
//! verbatim at inference — so a prediction made later sees exactly the
//! encoding the model was trained on.
//!
//! Encoding, per input column:
//! - **numeric** → one slot, standardized with the train mean/std. A missing
//!   value takes the mean (standardized 0); when the training split had any
//!   missing values, a second `<col> missing` slot (0/1) records that it was.
//! - **categorical** → one-hot over the train split's categories. A missing
//!   value is its own `(missing)` category when training saw one; a category
//!   training never saw encodes as all zeros.
//!
//! Columns that are neither (free text, IDs: too many distinct strings) are
//! left out and named in the run log, never guessed at.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Category label for a missing categorical value.
pub const MISSING: &str = "(missing)";

/// Above this many distinct strings a non-numeric column is treated as free
/// text / an ID and dropped, unless `[data] categorical` names it.
pub const MAX_CATEGORIES: usize = 50;

/// An integer-valued target with at most this many distinct values is read
/// as class ids under `task = "auto"` (the seeded starter datasets use small
/// integer ids); anything else numeric is a regression target.
const MAX_INT_CLASSES: usize = 20;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    Classification,
    Regression,
}

impl Task {
    pub fn as_str(self) -> &'static str {
        match self {
            Task::Classification => "classification",
            Task::Regression => "regression",
        }
    }
}

/// True for the spellings spreadsheets and exports use for "no value".
pub fn is_missing(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "" | "na" | "n/a" | "nan" | "null" | "none" | "?" | "-"
    )
}

fn parse_num(v: &str) -> Option<f64> {
    v.trim().parse::<f64>().ok().filter(|f| f.is_finite())
}

// ── CSV ───────────────────────────────────────────────────────────────

/// A parsed CSV: header plus rows, every row padded/truncated to the header
/// width so column indexing never goes out of bounds.
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// RFC 4180 CSV: quoted fields, `""` escapes, commas and newlines inside
/// quotes, CRLF line ends, a UTF-8 BOM. Blank lines are skipped.
pub fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut rows = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();
    let mut push_row = |row: &mut Vec<String>, field: &mut String| {
        row.push(std::mem::take(field));
        let r = std::mem::take(row);
        if !(r.len() == 1 && r[0].trim().is_empty()) {
            rows.push(r);
        }
    };
    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                '"' => in_quotes = false,
                _ => field.push(c),
            }
            continue;
        }
        match c {
            '"' if field.trim().is_empty() => {
                field.clear();
                in_quotes = true;
            }
            ',' => row.push(std::mem::take(&mut field)),
            '\r' => {}
            '\n' => push_row(&mut row, &mut field),
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        push_row(&mut row, &mut field);
    }
    rows
}

pub fn read_table(path: &Path) -> std::io::Result<Table> {
    let text = std::fs::read_to_string(path)?;
    let mut all = parse_csv(&text);
    if all.is_empty() {
        return Err(err(format!("{} is empty", path.display())));
    }
    let header: Vec<String> = all
        .remove(0)
        .into_iter()
        .map(|h| h.trim().to_string())
        .collect();
    let width = header.len();
    let rows: Vec<Vec<String>> = all
        .into_iter()
        .map(|mut r| {
            r.resize(width, String::new());
            r.into_iter().map(|v| v.trim().to_string()).collect()
        })
        .collect();
    if rows.is_empty() {
        return Err(err(format!("no data rows in {}", path.display())));
    }
    Ok(Table { header, rows })
}

pub fn err(msg: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

// ── column typing ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Numeric,
    Categorical,
    /// Dropped: free text / an ID, entirely empty, or listed in `ignore`.
    Skipped,
}

/// Decide a column's kind from its values. `force_cat` is `[data]
/// categorical` — numbers that are really labels (zip codes, ratings).
pub fn infer_kind(values: &[&str], force_cat: bool) -> Kind {
    let present: Vec<&str> = values.iter().copied().filter(|v| !is_missing(v)).collect();
    if present.is_empty() {
        return Kind::Skipped;
    }
    if force_cat {
        return Kind::Categorical;
    }
    if present.iter().all(|v| parse_num(v).is_some()) {
        return Kind::Numeric;
    }
    let distinct: BTreeSet<&str> = present.iter().copied().collect();
    if distinct.len() <= MAX_CATEGORIES {
        Kind::Categorical
    } else {
        Kind::Skipped
    }
}

/// `auto` picks classification for text labels and for a handful of integer
/// ids, regression for any other numeric target.
pub fn infer_task(values: &[&str]) -> Task {
    let present: Vec<&str> = values.iter().copied().filter(|v| !is_missing(v)).collect();
    let nums: Option<Vec<f64>> = present.iter().map(|v| parse_num(v)).collect();
    let Some(nums) = nums else {
        return Task::Classification;
    };
    let all_int = nums.iter().all(|v| v.fract() == 0.0);
    let distinct: BTreeSet<i64> = nums.iter().map(|v| *v as i64).collect();
    if all_int && distinct.len() <= MAX_INT_CLASSES {
        Task::Classification
    } else {
        Task::Regression
    }
}

// ── schema ────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Column {
    Numeric {
        name: String,
        mean: f32,
        std: f32,
        /// Adds a 0/1 "was missing" slot after the value slot.
        #[serde(default)]
        missing_flag: bool,
    },
    Categorical {
        name: String,
        categories: Vec<String>,
    },
}

impl Column {
    pub fn name(&self) -> &str {
        match self {
            Column::Numeric { name, .. } | Column::Categorical { name, .. } => name,
        }
    }

    fn width(&self) -> usize {
        match self {
            Column::Numeric { missing_flag, .. } => 1 + usize::from(*missing_flag),
            Column::Categorical { categories, .. } => categories.len(),
        }
    }
}

/// A request value, from a CSV cell or a JSON request.
pub enum Raw<'a> {
    Num(f64),
    Text(&'a str),
    Missing,
}

impl<'a> Raw<'a> {
    pub fn from_cell(v: &'a str) -> Self {
        if is_missing(v) {
            Raw::Missing
        } else {
            Raw::Text(v)
        }
    }

    pub fn from_json(v: Option<&'a serde_json::Value>) -> Self {
        match v {
            Some(serde_json::Value::Number(n)) => n.as_f64().map_or(Raw::Missing, Raw::Num),
            Some(serde_json::Value::String(s)) if !is_missing(s) => Raw::Text(s),
            Some(serde_json::Value::Bool(b)) => Raw::Text(if *b { "true" } else { "false" }),
            _ => Raw::Missing,
        }
    }

    fn as_num(&self) -> Option<f64> {
        match self {
            Raw::Num(n) => Some(*n),
            Raw::Text(s) => parse_num(s),
            Raw::Missing => None,
        }
    }

    fn as_text(&self) -> Option<String> {
        match self {
            Raw::Num(n) => Some(fmt_num(*n)),
            Raw::Text(s) => Some(s.trim().to_string()),
            Raw::Missing => None,
        }
    }
}

/// `3.0` → "3", so a JSON number matches a CSV category spelled "3".
fn fmt_num(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        n.to_string()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Schema {
    pub columns: Vec<Column>,
}

impl Schema {
    /// An all-numeric schema from a pre-0.9 checkpoint's `features` /
    /// `mean` / `std`, so older runs keep serving and exporting.
    pub fn legacy(features: &[String], mean: &[f32], std: &[f32]) -> Self {
        Schema {
            columns: features
                .iter()
                .enumerate()
                .map(|(i, name)| Column::Numeric {
                    name: name.clone(),
                    mean: mean.get(i).copied().unwrap_or(0.0),
                    std: std.get(i).copied().unwrap_or(1.0),
                    missing_flag: false,
                })
                .collect(),
        }
    }

    /// The model's input width.
    pub fn width(&self) -> usize {
        self.columns.iter().map(Column::width).sum()
    }

    /// The raw input names a request supplies (one per column).
    pub fn input_names(&self) -> Vec<String> {
        self.columns.iter().map(|c| c.name().to_string()).collect()
    }

    /// One name per encoded slot: `age`, `age missing`, `color=red`, …
    pub fn encoded_names(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(self.width());
        for c in &self.columns {
            match c {
                Column::Numeric {
                    name, missing_flag, ..
                } => {
                    out.push(name.clone());
                    if *missing_flag {
                        out.push(format!("{name} missing"));
                    }
                }
                Column::Categorical { name, categories } => {
                    out.extend(categories.iter().map(|k| format!("{name}={k}")));
                }
            }
        }
        out
    }

    /// Per-slot (mean, std) over the encoded vector: the numeric stats in
    /// value slots, (0, 1) everywhere else. The ONNX export bakes these in as
    /// `Sub`/`Div`, so its input is [`Self::encode_raw`]'s output.
    pub fn encoded_stats(&self) -> (Vec<f32>, Vec<f32>) {
        let mut mean = Vec::with_capacity(self.width());
        let mut std = Vec::with_capacity(self.width());
        for c in &self.columns {
            match c {
                Column::Numeric {
                    mean: m,
                    std: s,
                    missing_flag,
                    ..
                } => {
                    mean.push(*m);
                    std.push(s.max(1e-8));
                    if *missing_flag {
                        mean.push(0.0);
                        std.push(1.0);
                    }
                }
                Column::Categorical { categories, .. } => {
                    mean.extend(std::iter::repeat_n(0.0, categories.len()));
                    std.extend(std::iter::repeat_n(1.0, categories.len()));
                }
            }
        }
        (mean, std)
    }

    /// The encoded vector before standardization: numeric values as given
    /// (missing → the train mean), one-hot and missing slots as 0/1.
    pub fn encode_raw<'a>(&self, get: impl Fn(&str) -> Raw<'a>) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.width());
        for c in &self.columns {
            let raw = get(c.name());
            match c {
                Column::Numeric {
                    mean, missing_flag, ..
                } => {
                    let v = raw.as_num();
                    out.push(v.map_or(*mean, |v| v as f32));
                    if *missing_flag {
                        out.push(if v.is_none() { 1.0 } else { 0.0 });
                    }
                }
                Column::Categorical { categories, .. } => {
                    let key = raw.as_text().unwrap_or_else(|| MISSING.to_string());
                    out.extend(categories.iter().map(|k| if *k == key { 1.0 } else { 0.0 }));
                }
            }
        }
        out
    }

    /// The model input: [`Self::encode_raw`] standardized slot by slot.
    pub fn encode<'a>(&self, get: impl Fn(&str) -> Raw<'a>) -> Vec<f32> {
        let (mean, std) = self.encoded_stats();
        self.encode_raw(get)
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v - mean[i]) / std[i])
            .collect()
    }

    /// A short "what the model reads" line for the run log.
    pub fn describe(&self) -> String {
        let (mut num, mut cat) = (0, 0);
        for c in &self.columns {
            match c {
                Column::Numeric { .. } => num += 1,
                Column::Categorical { .. } => cat += 1,
            }
        }
        format!(
            "{num} numeric + {cat} categorical columns → {} inputs",
            self.width()
        )
    }
}

// ── fitting ───────────────────────────────────────────────────────────

/// Column choices that come from `[data]` in train.toml.
#[derive(Default)]
pub struct ColumnPrefs<'a> {
    pub categorical: &'a [String],
    pub ignore: &'a [String],
}

/// The input columns to use, with their kinds, plus the names left out (and
/// why) for the run log. `target` is never an input.
pub fn plan_columns(
    table: &Table,
    target_idx: usize,
    prefs: &ColumnPrefs,
) -> (Vec<(usize, Kind)>, Vec<String>) {
    let mut used = Vec::new();
    let mut notes = Vec::new();
    for (i, name) in table.header.iter().enumerate() {
        if i == target_idx {
            continue;
        }
        if prefs.ignore.iter().any(|n| n == name) {
            notes.push(format!("{name} (ignored in train.toml)"));
            continue;
        }
        let values: Vec<&str> = table.rows.iter().map(|r| r[i].as_str()).collect();
        let force = prefs.categorical.iter().any(|n| n == name);
        match infer_kind(&values, force) {
            Kind::Skipped => notes.push(format!(
                "{name} (empty, or more than {MAX_CATEGORIES} distinct text values)"
            )),
            k => used.push((i, k)),
        }
    }
    (used, notes)
}

/// Fit the schema on the training rows: numeric mean/std over present
/// values, the categories (sorted) seen in training, and whether a missing
/// slot is needed.
pub fn fit_schema(
    header: &[String],
    plan: &[(usize, Kind)],
    train_rows: &[&Vec<String>],
) -> Schema {
    let columns = plan
        .iter()
        .map(|&(i, kind)| {
            let name = header[i].clone();
            let cells = train_rows.iter().map(|r| r[i].as_str());
            match kind {
                Kind::Numeric => {
                    let vals: Vec<f64> = cells.clone().filter_map(parse_num).collect();
                    let missing = cells.clone().any(is_missing);
                    let n = vals.len().max(1) as f64;
                    let mean = vals.iter().sum::<f64>() / n;
                    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                    Column::Numeric {
                        name,
                        mean: mean as f32,
                        std: (var.sqrt() as f32).max(1e-8),
                        missing_flag: missing,
                    }
                }
                _ => {
                    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
                    for c in cells {
                        let key = if is_missing(c) {
                            MISSING.to_string()
                        } else {
                            c.to_string()
                        };
                        seen.insert(key, ());
                    }
                    Column::Categorical {
                        name,
                        categories: seen.into_keys().collect(),
                    }
                }
            }
        })
        .collect();
    Schema { columns }
}

/// Inverse-frequency class weights, `n / (k · n_c)`, over the training
/// labels. A class absent from training gets weight 1.
pub fn balanced_weights(labels: &[usize], n_classes: usize) -> Vec<f32> {
    let mut counts = vec![0usize; n_classes];
    for &l in labels {
        if l < n_classes {
            counts[l] += 1;
        }
    }
    let n = labels.len() as f32;
    counts
        .iter()
        .map(|&c| {
            if c == 0 {
                1.0
            } else {
                n / (n_classes as f32 * c as f32)
            }
        })
        .collect()
}

/// Largest class count over the smallest non-zero one.
pub fn imbalance_ratio(labels: &[usize], n_classes: usize) -> f32 {
    let mut counts = vec![0usize; n_classes];
    for &l in labels {
        if l < n_classes {
            counts[l] += 1;
        }
    }
    let nz: Vec<usize> = counts.into_iter().filter(|&c| c > 0).collect();
    match (nz.iter().max(), nz.iter().min()) {
        (Some(&hi), Some(&lo)) => hi as f32 / lo as f32,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_handles_quotes_crlf_and_bom() {
        let rows =
            parse_csv("\u{feff}a,b\r\n\"x, y\",\"say \"\"hi\"\"\"\r\n\r\n1,\"multi\nline\"\n");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec!["a", "b"]);
        assert_eq!(rows[1], vec!["x, y", "say \"hi\""]);
        assert_eq!(rows[2], vec!["1", "multi\nline"]);
    }

    #[test]
    fn kinds_numeric_categorical_and_skipped() {
        assert_eq!(infer_kind(&["1", "2.5", "", "NA"], false), Kind::Numeric);
        assert_eq!(
            infer_kind(&["red", "blue", "red"], false),
            Kind::Categorical
        );
        assert_eq!(infer_kind(&["90210", "10001"], true), Kind::Categorical);
        assert_eq!(infer_kind(&["", "n/a"], false), Kind::Skipped);
        let ids: Vec<String> = (0..60).map(|i| format!("id-{i}")).collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        assert_eq!(infer_kind(&refs, false), Kind::Skipped);
    }

    #[test]
    fn task_auto_detection() {
        assert_eq!(infer_task(&["cat", "dog"]), Task::Classification);
        assert_eq!(infer_task(&["0", "1", "2", "1"]), Task::Classification);
        assert_eq!(infer_task(&["12.5", "3.1", "7"]), Task::Regression);
        let many: Vec<String> = (0..40).map(|i| i.to_string()).collect();
        let refs: Vec<&str> = many.iter().map(String::as_str).collect();
        assert_eq!(infer_task(&refs), Task::Regression);
    }

    fn table() -> Table {
        let rows = vec![
            vec!["1".into(), "red".into(), "10".into()],
            vec!["".into(), "blue".into(), "20".into()],
            vec!["3".into(), "".into(), "30".into()],
        ];
        Table {
            header: vec!["n".into(), "color".into(), "y".into()],
            rows,
        }
    }

    #[test]
    fn schema_fits_imputes_and_one_hots() {
        let t = table();
        let (plan, notes) = plan_columns(&t, 2, &ColumnPrefs::default());
        assert!(notes.is_empty());
        let rows: Vec<&Vec<String>> = t.rows.iter().collect();
        let s = fit_schema(&t.header, &plan, &rows);
        assert_eq!(
            s.encoded_names(),
            vec![
                "n",
                "n missing",
                "color=(missing)",
                "color=blue",
                "color=red"
            ]
        );
        // Missing numeric → mean (2.0) + flag; text category one-hot.
        let raw = s.encode_raw(|c| match c {
            "color" => Raw::Text("red"),
            _ => Raw::Missing,
        });
        assert_eq!(raw, vec![2.0, 1.0, 0.0, 0.0, 1.0]);
        // Standardized: the imputed mean is exactly 0.
        let enc = s.encode(|c| match c {
            "color" => Raw::Text("green"), // unseen → all zeros
            _ => Raw::Missing,
        });
        assert_eq!(enc, vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        // A JSON number matches a category spelled as an integer.
        assert_eq!(Raw::Num(3.0).as_text().as_deref(), Some("3"));
    }

    #[test]
    fn schema_round_trips_through_json() {
        let t = table();
        let (plan, _) = plan_columns(&t, 2, &ColumnPrefs::default());
        let rows: Vec<&Vec<String>> = t.rows.iter().collect();
        let s = fit_schema(&t.header, &plan, &rows);
        let back: Schema = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn ignore_list_drops_a_column() {
        let t = table();
        let ignore = vec!["color".to_string()];
        let (plan, notes) = plan_columns(
            &t,
            2,
            &ColumnPrefs {
                categorical: &[],
                ignore: &ignore,
            },
        );
        assert_eq!(plan.len(), 1);
        assert_eq!(notes.len(), 1);
    }

    #[test]
    fn class_weights_favor_the_rare_class() {
        let labels = [0, 0, 0, 1];
        let w = balanced_weights(&labels, 2);
        assert!(w[1] > w[0]);
        assert!((imbalance_ratio(&labels, 2) - 3.0).abs() < 1e-6);
    }
}
