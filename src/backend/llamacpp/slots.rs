//! Parsing of the `/slots` endpoint.
//!
//! The response is a JSON array of slot objects. llama.cpp has changed the
//! slot schema between releases:
//!
//! - `next_token` was a single object in older builds and is an array of one
//!   object in current builds; both forms are accepted.
//! - current builds report prompt occupancy as `n_prompt_tokens` (the prompt
//!   buffer size) and no longer emit a top-level `n_tokens`; older builds
//!   emit `n_tokens` and no `n_prompt_tokens`.
//!
//! Example (current build):
//! ```json
//! {
//!   "id": 0,
//!   "n_ctx": 237568,
//!   "speculative": true,
//!   "is_processing": false,
//!   "id_task": 123,
//!   "n_prompt_tokens": 58029,
//!   "n_prompt_tokens_processed": 0,
//!   "n_prompt_tokens_cache": 0,
//!   "params": { "...": "..." },
//!   "next_token": [
//!     {
//!       "has_next_token": false,
//!       "has_new_line": false,
//!       "n_remain": -1,
//!       "n_decoded": 0
//!     }
//!   ]
//! }
//! ```
//!
//! Only fields llamatop actually consumes are deserialized; everything else
//! (`params` and any future field) is ignored. Fields that can be absent
//! (idle slot, older versions) are `Option`. The `prompt`/`generated` fields
//! (only present with slots debugging) are intentionally not deserialized
//! into any type we keep: we never store prompt or completion text.

use serde::{Deserialize, Serialize};

/// One raw slot as reported by the server (wire format only).
///
/// Prompt/completion text (`prompt`, `generated`) is deliberately not a field
/// here, so serializing this type can never leak request content.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct RawSlot {
    pub id: u32,
    pub n_ctx: Option<u64>,
    /// Total context occupancy. Older builds only; current builds omit it.
    pub n_tokens: Option<u64>,
    #[serde(rename = "is_processing")]
    pub is_processing: bool,
    pub speculative: bool,
    #[serde(rename = "id_task")]
    pub id_task: Option<u64>,
    pub n_prompt_tokens: Option<u64>,
    pub n_prompt_tokens_processed: Option<u64>,
    pub n_prompt_tokens_cache: Option<u64>,
    /// Older builds send a single object; current builds send an array of
    /// one object. Both are accepted.
    pub next_token: Option<OneOrMany<RawNextToken>>,
}

/// A field that some llama.cpp releases send as a single object and later
/// releases changed into an array (e.g. `next_token`). Only the first
/// element is meaningful; the rest (if any) are ignored.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    /// The first element, whichever form the server used.
    pub fn first(&self) -> Option<&T> {
        match self {
            OneOrMany::One(v) => Some(v),
            OneOrMany::Many(v) => v.first(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct RawNextToken {
    pub has_next_token: bool,
    pub has_new_line: bool,
    pub n_remain: Option<i64>,
    pub n_decoded: Option<u64>,
}

/// Parse the raw `/slots` body. Unknown fields are ignored; a non-array body
/// is a parse error.
pub fn parse_slots(body: &str) -> Result<Vec<RawSlot>, String> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("invalid JSON: {e}"))?;
    let arr = value.as_array().ok_or_else(|| "expected a JSON array of slots".to_string())?;
    let mut slots = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let slot: RawSlot =
            serde_json::from_value(item.clone()).map_err(|e| format!("slot {i}: {e}"))?;
        slots.push(slot);
    }
    Ok(slots)
}

/// Convenience: the generated-token counter for a slot (None when idle/absent).
pub fn decoded_tokens(slot: &RawSlot) -> Option<u64> {
    slot.next_token.as_ref().and_then(|nt| nt.first()).and_then(|nt| nt.n_decoded)
}

/// Convenience: current context occupancy for a slot, when derivable.
///
/// `n_tokens` is the total occupancy on older builds and takes precedence.
/// Current builds omit it and report the prompt buffer size as
/// `n_prompt_tokens` instead. The `*_processed`/`*_cache` counters are
/// sub-counts of the prompt buffer, not additional occupancy: adding them
/// would double-count, so they are never summed in here.
pub fn context_used(slot: &RawSlot) -> Option<u64> {
    slot.n_tokens.or(slot.n_prompt_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_processing_slot() {
        let body = r#"[
            {
                "id": 0,
                "n_ctx": 4096,
                "speculative": false,
                "is_processing": true,
                "id_task": 123,
                "n_prompt_tokens": 100,
                "n_prompt_tokens_processed": 100,
                "n_prompt_tokens_cache": 0,
                "params": {"n_predict": -1, "temperature": 0.8},
                "next_token": {"has_next_token": true, "has_new_line": false, "n_remain": -1, "n_decoded": 42}
            }
        ]"#;
        let slots = parse_slots(body).unwrap();
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].id, 0);
        assert!(slots[0].is_processing);
        assert_eq!(decoded_tokens(&slots[0]), Some(42));
        // No n_tokens on this shape: occupancy is the prompt buffer size.
        // processed/cache are sub-counts and must not be added.
        assert_eq!(context_used(&slots[0]), Some(100));
    }

    #[test]
    fn next_token_object_does_not_break_older_responses() {
        let body =
            r#"[{"id": 0, "n_ctx": 4096, "is_processing": true, "next_token": {"n_decoded": 7}}]"#;
        let slots = parse_slots(body).unwrap();
        assert_eq!(decoded_tokens(&slots[0]), Some(7));
    }

    #[test]
    fn next_token_array_of_multiple_uses_first_element() {
        let body = r#"[{"id": 0, "is_processing": true, "next_token": [
            {"has_next_token": true, "n_remain": -1, "n_decoded": 7},
            {"has_next_token": false, "n_remain": -1, "n_decoded": 99}
        ]}]"#;
        let slots = parse_slots(body).unwrap();
        assert_eq!(decoded_tokens(&slots[0]), Some(7));
    }

    #[test]
    fn missing_next_token_parses() {
        let body = r#"[{"id": 0, "is_processing": false}]"#;
        let slots = parse_slots(body).unwrap();
        assert!(slots[0].next_token.is_none());
        assert_eq!(decoded_tokens(&slots[0]), None);
    }

    #[test]
    fn parses_idle_slot_with_optional_fields_missing() {
        let body = r#"[{"id": 1, "n_ctx": 8192, "speculative": true, "is_processing": false}]"#;
        let slots = parse_slots(body).unwrap();
        assert_eq!(slots[0].id_task, None);
        assert_eq!(decoded_tokens(&slots[0]), None);
        assert_eq!(context_used(&slots[0]), None);
    }

    #[test]
    fn empty_array_is_ok() {
        assert!(parse_slots("[]").unwrap().is_empty());
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let body = r#"[{"id": 0, "is_processing": false, "some_future_field": {"x": 1}}]"#;
        assert!(parse_slots(body).is_ok());
    }

    #[test]
    fn non_array_body_is_an_error() {
        assert!(parse_slots(r#"{"id":0}"#).is_err());
    }

    #[test]
    fn null_fields_are_tolerated() {
        let body = r#"[{"id": 0, "is_processing": false, "n_ctx": null, "next_token": null}]"#;
        let slots = parse_slots(body).unwrap();
        assert_eq!(slots[0].n_ctx, None);
        assert_eq!(decoded_tokens(&slots[0]), None);
    }

    #[test]
    fn invalid_json_is_an_error() {
        assert!(parse_slots("{nope").is_err());
    }

    /// Anonymized capture of a live llama-server `/slots` response (current
    /// schema): top-level array, `next_token` as an array, `params` with
    /// arbitrary sampling fields, no top-level `n_tokens`. This is the
    /// regression that produced "Slots response could not be parsed".
    #[test]
    fn live_anonymized_slots_fixture_parses() {
        let body = include_str!("../../../fixtures/slots_live_anon.json");
        let slots = parse_slots(body).expect("the current llama.cpp slot schema must parse");
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].id, 0);
        assert_eq!(slots[0].n_ctx, Some(237568));
        assert!(!slots[0].is_processing);
        assert_eq!(slots[0].n_prompt_tokens, Some(58029));
        assert_eq!(decoded_tokens(&slots[0]), Some(0));
        // Unknown params fields must not be retained (no sampling state,
        // no prompt-related values).
        let raw = serde_json::to_string(&slots[0]).unwrap();
        assert!(!raw.contains("generation_prompt"));
        assert!(!raw.contains("chat_format"));
        assert!(!raw.contains("samplers"));
    }

    #[test]
    fn fixture_slots_idle_parses() {
        let body = include_str!("../../../fixtures/slots_idle.json");
        let slots = parse_slots(body).unwrap();
        assert_eq!(slots.len(), 2);
        assert!(!slots[0].is_processing);
        assert!(!slots[1].is_processing);
        assert_eq!(decoded_tokens(&slots[0]), None);
    }

    #[test]
    fn fixture_slots_processing_parses() {
        let body = include_str!("../../../fixtures/slots_processing.json");
        let slots = parse_slots(body).unwrap();
        assert!(slots[0].is_processing);
        assert_eq!(slots[0].id_task, Some(135));
        assert_eq!(decoded_tokens(&slots[0]), Some(0));
        assert_eq!(context_used(&slots[0]), Some(1000));
    }

    #[test]
    fn fixture_slots_decode_parses() {
        let body = include_str!("../../../fixtures/slots_decode.json");
        let slots = parse_slots(body).unwrap();
        assert!(slots[0].is_processing);
        assert_eq!(decoded_tokens(&slots[0]), Some(624));
        // No n_tokens in this fixture: occupancy is the prompt buffer size,
        // with the processed sub-count (1000) not added on top.
        assert_eq!(context_used(&slots[0]), Some(1000));
    }

    #[test]
    fn missing_n_tokens_falls_back_to_n_prompt_tokens() {
        let body =
            r#"[{"id": 0, "is_processing": false, "n_ctx": 8192, "n_prompt_tokens": 58029}]"#;
        let slots = parse_slots(body).unwrap();
        assert_eq!(context_used(&slots[0]), Some(58029));
    }

    #[test]
    fn n_tokens_takes_precedence_over_n_prompt_tokens() {
        let body = r#"[{"id": 0, "is_processing": false, "n_ctx": 8192, "n_tokens": 123, "n_prompt_tokens": 100}]"#;
        let slots = parse_slots(body).unwrap();
        assert_eq!(context_used(&slots[0]), Some(123));
    }

    #[test]
    fn missing_both_token_fields_stays_unavailable() {
        let body = r#"[{"id": 0, "is_processing": false, "n_ctx": 8192}]"#;
        let slots = parse_slots(body).unwrap();
        assert_eq!(context_used(&slots[0]), None);
    }

    #[test]
    fn minimal_slot_object_parses() {
        let body = r#"[{"id": 0}]"#;
        let slots = parse_slots(body).unwrap();
        assert_eq!(slots[0].id, 0);
        assert_eq!(slots[0].n_ctx, None);
        assert_eq!(context_used(&slots[0]), None);
        assert_eq!(decoded_tokens(&slots[0]), None);
    }

    #[test]
    fn successful_empty_array_differs_from_parse_failed() {
        // An empty array is a successful parse of zero slots (the endpoint
        // works); it must not be reported the same way as a parse failure.
        assert!(parse_slots("[]").is_ok_and(|s| s.is_empty()));
        assert!(parse_slots(r#"[{"id": 0, "n_ctx": "not-a-number"}]"#).is_err());
    }

    #[test]
    fn prompt_and_generated_fields_are_not_retained() {
        // Even if the server includes prompt/generated (debug mode), we parse
        // them as unknown fields and keep no text at all.
        let body = r#"[{"id": 0, "is_processing": false, "prompt": "SECRET PROMPT", "generated": "SECRET COMPLETION"}]"#;
        let slots = parse_slots(body).unwrap();
        let raw = serde_json::to_string(&slots).unwrap();
        assert!(!raw.contains("SECRET"));
    }
}
