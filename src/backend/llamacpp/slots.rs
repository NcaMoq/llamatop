//! Parsing of the `/slots` endpoint.
//!
//! Current llama.cpp response is a JSON array of slot objects:
//! ```json
//! {
//!   "id": 0,
//!   "n_ctx": 4096,
//!   "speculative": false,
//!   "is_processing": true,
//!   "id_task": 123,
//!   "n_prompt_tokens": 100,
//!   "n_prompt_tokens_processed": 97,
//!   "n_prompt_tokens_cache": 0,
//!   "params": { "...": "..." },
//!   "next_token": {
//!     "has_next_token": true,
//!     "has_new_line": false,
//!     "n_remain": -1,
//!     "n_decoded": 42
//!   }
//! }
//! ```
//!
//! Fields that can be absent (idle slot, older versions) are `Option`. The
//! `prompt`/`generated` fields (only present with slots debugging) are
//! intentionally not deserialized into any type we keep: we never store prompt
//! or completion text.

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
    #[serde(rename = "is_processing")]
    pub is_processing: bool,
    pub speculative: bool,
    #[serde(rename = "id_task")]
    pub id_task: Option<u64>,
    pub n_prompt_tokens: Option<u64>,
    pub n_prompt_tokens_processed: Option<u64>,
    pub n_prompt_tokens_cache: Option<u64>,
    pub next_token: Option<RawNextToken>,
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
    slot.next_token.as_ref().and_then(|nt| nt.n_decoded)
}

/// Convenience: current context occupancy for a slot, when derivable.
pub fn context_used(slot: &RawSlot) -> Option<u64> {
    // Context used = prompt tokens in context + generated tokens.
    let prompt = slot.n_prompt_tokens?;
    let decoded = decoded_tokens(slot).unwrap_or(0);
    Some(prompt + decoded)
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
        assert_eq!(context_used(&slots[0]), Some(142));
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
        // Context used = prompt tokens + decoded.
        assert_eq!(context_used(&slots[0]), Some(1624));
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
