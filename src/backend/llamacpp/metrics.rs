//! Parsing of the `/metrics` endpoint (Prometheus text format 0.0.4).
//!
//! Metric names observed on current llama.cpp (prefix `llamacpp:`):
//!
//! counters:  prompt_tokens_total, prompt_tokens_cached_total,
//!            prompt_seconds_total, tokens_predicted_total,
//!            tokens_predicted_seconds_total, n_decode_total, n_tokens_max,
//!            spec_decode_num_draft_tokens_total,
//!            spec_decode_num_accepted_tokens_total, spec_decode_num_drafts_total,
//!            spec_decode_num_accepted_tokens_per_pos_total{position="N"}
//! gauges:    prompt_tokens_seconds, predicted_tokens_seconds,
//!            requests_processing, requests_deferred, n_busy_slots_per_decode
//!
//! Parsing is deliberately tolerant:
//! - comments (`# HELP` / `# TYPE`) and unknown metrics are ignored
//! - a non-finite sample value (NaN, Inf) marks that metric as missing (None)
//! - a malformed line is skipped, not fatal
//!
//! Raw metric names never leak past this module; `LlamaCppRawMetrics` is the
//! normalized wire-level view, and `BackendSnapshot` is the domain view.

use std::collections::BTreeMap;

use crate::domain::SpeculativeStats;

/// Normalized wire-level metrics (values as reported; units preserved).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LlamaCppRawMetrics {
    pub prompt_tokens_total: Option<f64>,
    pub prompt_tokens_cached_total: Option<f64>,
    pub prompt_seconds_total: Option<f64>,
    pub tokens_predicted_total: Option<f64>,
    pub tokens_predicted_seconds_total: Option<f64>,
    pub n_decode_total: Option<f64>,
    pub n_tokens_max: Option<f64>,
    pub spec_decode_num_draft_tokens_total: Option<f64>,
    pub spec_decode_num_accepted_tokens_total: Option<f64>,
    pub spec_decode_num_drafts_total: Option<f64>,
    /// Accepted tokens per draft position (index = position).
    pub spec_decode_num_accepted_tokens_per_pos: Vec<Option<f64>>,
    pub prompt_tokens_seconds: Option<f64>,
    pub predicted_tokens_seconds: Option<f64>,
    pub requests_processing: Option<f64>,
    pub requests_deferred: Option<f64>,
    pub n_busy_slots_per_decode: Option<f64>,
}

impl LlamaCppRawMetrics {
    /// Convert a (possibly fractional) metric value to an integer counter.
    /// Non-finite or negative values become None (never a fake 0).
    pub fn counter(&self, value: Option<f64>) -> Option<u64> {
        let v = value?;
        if !v.is_finite() || v < 0.0 {
            return None;
        }
        Some(v as u64)
    }

    /// Convert a gauge value to a non-negative integer where appropriate.
    pub fn gauge_u64(&self, value: Option<f64>) -> Option<u64> {
        let v = value?;
        if !v.is_finite() || v < 0.0 {
            return None;
        }
        Some(v as u64)
    }

    /// Keep a floating-point value only when it is finite and non-negative;
    /// otherwise `None` (missing, not 0).
    pub fn finite(&self, value: Option<f64>) -> Option<f64> {
        let v = value?;
        if v.is_finite() && v >= 0.0 {
            Some(v)
        } else {
            None
        }
    }

    /// Fill the speculative-decoding section of a snapshot.
    pub fn into_speculative_stats(&self) -> SpeculativeStats {
        SpeculativeStats {
            draft_tokens_total: self.counter(self.spec_decode_num_draft_tokens_total),
            accepted_tokens_total: self.counter(self.spec_decode_num_accepted_tokens_total),
            drafts_total: self.counter(self.spec_decode_num_drafts_total),
        }
    }
}

/// Parsed Prometheus text: unlabeled samples plus labeled samples keyed by the
/// `position` label (the only label the llama.cpp server uses).
#[derive(Debug, Clone, Default)]
struct Parsed {
    samples: BTreeMap<String, f64>,
    per_position: BTreeMap<String, BTreeMap<u64, f64>>,
}

/// Parse Prometheus exposition text into samples.
fn parse_prometheus(text: &str) -> Parsed {
    let mut parsed = Parsed::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        parse_sample(line, &mut parsed);
    }
    parsed
}

fn parse_sample(line: &str, parsed: &mut Parsed) {
    // Shape: name  value            or   name{labels}  value
    let (name_part, value_part) = match line.rsplit_once(' ') {
        Some((n, v)) => (n, v),
        None => return,
    };
    let value: f64 = match value_part.parse::<f64>() {
        Ok(v) if v.is_finite() => v,
        // NaN / +-Inf / garbage: treat as missing, keep going.
        _ => return,
    };

    let (name, labels) = match name_part.find('{') {
        Some(idx) => (&name_part[..idx], &name_part[idx + 1..name_part.len() - 1]),
        None => (name_part, ""),
    };
    if name.is_empty() {
        return;
    }

    if labels.is_empty() {
        // Last sample wins (Prometheus does not repeat unlabeled series).
        parsed.samples.insert(name.to_string(), value);
        return;
    }

    // Extract the position label if present: position="3".
    let mut position: Option<u64> = None;
    for (key, value) in parse_labels(labels) {
        if key == "position" {
            position = value.parse().ok();
        }
    }
    if let Some(pos) = position {
        parsed.per_position.entry(name.to_string()).or_default().insert(pos, value);
    }
    // Samples with other/unknown labels are ignored.
}

fn parse_labels(input: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for part in input.split(',') {
        let part = part.trim();
        if let Some(eq) = part.find('=') {
            let key = part[..eq].trim().to_string();
            let raw_val = part[eq + 1..].trim();
            let val = raw_val
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .unwrap_or(raw_val)
                .to_string();
            out.push((key, val));
        }
    }
    out
}

const SPEC_PER_POS: &str = "llamacpp:spec_decode_num_accepted_tokens_per_pos_total";

impl From<Parsed> for LlamaCppRawMetrics {
    fn from(p: Parsed) -> Self {
        let get = |name: &str| -> Option<f64> { p.samples.get(name).copied() };

        let mut per_pos: Vec<Option<f64>> = Vec::new();
        if let Some(map) = p.per_position.get(SPEC_PER_POS) {
            let max_pos = map.keys().max().copied().unwrap_or(0) as usize;
            per_pos = (0..=max_pos).map(|i| map.get(&(i as u64)).copied()).collect();
        }

        Self {
            prompt_tokens_total: get("llamacpp:prompt_tokens_total"),
            prompt_tokens_cached_total: get("llamacpp:prompt_tokens_cached_total"),
            prompt_seconds_total: get("llamacpp:prompt_seconds_total"),
            tokens_predicted_total: get("llamacpp:tokens_predicted_total"),
            tokens_predicted_seconds_total: get("llamacpp:tokens_predicted_seconds_total"),
            n_decode_total: get("llamacpp:n_decode_total"),
            n_tokens_max: get("llamacpp:n_tokens_max"),
            spec_decode_num_draft_tokens_total: get("llamacpp:spec_decode_num_draft_tokens_total"),
            spec_decode_num_accepted_tokens_total: get(
                "llamacpp:spec_decode_num_accepted_tokens_total",
            ),
            spec_decode_num_drafts_total: get("llamacpp:spec_decode_num_drafts_total"),
            spec_decode_num_accepted_tokens_per_pos: per_pos,
            prompt_tokens_seconds: get("llamacpp:prompt_tokens_seconds"),
            predicted_tokens_seconds: get("llamacpp:predicted_tokens_seconds"),
            requests_processing: get("llamacpp:requests_processing"),
            requests_deferred: get("llamacpp:requests_deferred"),
            n_busy_slots_per_decode: get("llamacpp:n_busy_slots_per_decode"),
        }
    }
}

/// Parse a full `/metrics` body into normalized wire metrics.
pub fn parse_metrics(body: &str) -> LlamaCppRawMetrics {
    parse_prometheus(body).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed, excluding cached tokens
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 1234
# HELP llamacpp:requests_processing Number of requests processing
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 2
llamacpp:requests_deferred 1
llamacpp:predicted_tokens_seconds 48.6
llamacpp:tokens_predicted_total 77
llamacpp:n_tokens_max 4096
# HELP unknown_metric some future metric
# TYPE unknown_metric counter
llamacpp:some_future_metric 99
"#;

    #[test]
    fn parses_known_metrics_and_ignores_unknown() {
        let m = parse_metrics(SAMPLE);
        assert_eq!(m.prompt_tokens_total, Some(1234.0));
        assert_eq!(m.requests_processing, Some(2.0));
        assert_eq!(m.requests_deferred, Some(1.0));
        assert!((m.predicted_tokens_seconds.unwrap() - 48.6).abs() < 1e-9);
        assert_eq!(m.tokens_predicted_total, Some(77.0));
        assert_eq!(m.n_tokens_max, Some(4096.0));
        // Unknown metric simply absent.
        assert!(m.spec_decode_num_drafts_total.is_none());
    }

    #[test]
    fn counter_conversion_rejects_negative_and_nan() {
        let m = LlamaCppRawMetrics { prompt_tokens_total: Some(-5.0), ..Default::default() };
        assert_eq!(m.counter(m.prompt_tokens_total), None);

        let m = LlamaCppRawMetrics { prompt_tokens_total: Some(f64::NAN), ..Default::default() };
        assert_eq!(m.counter(m.prompt_tokens_total), None);

        let m = LlamaCppRawMetrics { prompt_tokens_total: Some(42.9), ..Default::default() };
        assert_eq!(m.counter(m.prompt_tokens_total), Some(42));
    }

    #[test]
    fn nan_and_inf_samples_are_missing() {
        let text = "llamacpp:prompt_tokens_total NaN\nllamacpp:requests_processing +Inf\n";
        let m = parse_metrics(text);
        assert_eq!(m.prompt_tokens_total, None);
        assert_eq!(m.requests_processing, None);
    }

    #[test]
    fn malformed_lines_are_skipped() {
        let text = "llamacpp:prompt_tokens_total 10\nthis is not a sample line\nllamacpp:requests_processing 3\n";
        let m = parse_metrics(text);
        assert_eq!(m.prompt_tokens_total, Some(10.0));
        assert_eq!(m.requests_processing, Some(3.0));
    }

    #[test]
    fn labeled_position_series_are_collected() {
        let text = r#"# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 5
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 3
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="3"} 1
"#;
        let m = parse_metrics(text);
        assert_eq!(
            m.spec_decode_num_accepted_tokens_per_pos,
            vec![Some(5.0), Some(3.0), None, Some(1.0)]
        );
    }

    #[test]
    fn empty_body_yields_all_none() {
        let m = parse_metrics("");
        assert_eq!(m, LlamaCppRawMetrics::default());
    }

    #[test]
    fn changed_metric_order_is_tolerated() {
        let a = parse_metrics("llamacpp:requests_processing 1\nllamacpp:prompt_tokens_total 5\n");
        let b = parse_metrics("llamacpp:prompt_tokens_total 5\nllamacpp:requests_processing 1\n");
        assert_eq!(a, b);
    }

    #[test]
    fn fixture_metrics_parses() {
        let body = include_str!("../../../fixtures/metrics.txt");
        let m = parse_metrics(body);
        assert_eq!(m.prompt_tokens_total, Some(12345.0));
        assert_eq!(m.tokens_predicted_total, Some(678.0));
        assert_eq!(m.n_tokens_max, Some(24576.0));
        assert_eq!(m.requests_processing, Some(2.0));
        assert_eq!(m.requests_deferred, Some(1.0));
        assert!((m.prompt_tokens_seconds.unwrap() - 1832.4).abs() < 1e-9);
        assert!((m.predicted_tokens_seconds.unwrap() - 48.6).abs() < 1e-9);
        assert_eq!(m.spec_decode_num_draft_tokens_total, Some(100.0));
        assert_eq!(m.spec_decode_num_accepted_tokens_total, Some(60.0));
        assert_eq!(m.spec_decode_num_drafts_total, Some(20.0));
        // per-position: position 0 and 1 present.
        assert_eq!(m.spec_decode_num_accepted_tokens_per_pos, vec![Some(55.0), Some(5.0)]);
        // Unknown future metric is ignored.
    }

    #[test]
    fn speculative_stats_mapping() {
        let m = LlamaCppRawMetrics {
            spec_decode_num_draft_tokens_total: Some(100.0),
            spec_decode_num_accepted_tokens_total: Some(60.0),
            spec_decode_num_drafts_total: Some(20.0),
            ..Default::default()
        };
        let stats = m.into_speculative_stats();
        assert_eq!(stats.draft_tokens_total, Some(100));
        assert_eq!(stats.accepted_tokens_total, Some(60));
        assert_eq!(stats.drafts_total, Some(20));
        // acceptance_rate = accepted / draft tokens = 60/100 = 0.6 (0.0..=1.0).
        assert!((stats.acceptance_rate().unwrap() - 0.6).abs() < 1e-9);
    }
}
