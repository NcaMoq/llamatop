//! Parsing of the `/props` endpoint.
//!
//! Top-level fields observed on current llama.cpp (all optional here so that
//! older/newer builds degrade gracefully):
//! default_generation_settings{params,n_ctx}, total_slots, model_alias,
//! model_ftype, model_path, modalities, media_marker, endpoint_slots,
//! endpoint_props, endpoint_metrics, ui, ui_settings, chat_template,
//! chat_template_caps, bos_token, eos_token, build_info, is_sleeping,
//! cors_proxy_enabled.
//!
//! Only monitoring-relevant fields are extracted. The chat template and
//! default generation parameters are deliberately not retained.

use serde::{Deserialize, Serialize};

/// Raw `/props` response (wire format only, monitoring subset).
///
/// Only monitoring-relevant fields are kept, so serializing this type can
/// never leak the chat template or generation parameters.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct RawProps {
    pub total_slots: Option<u64>,
    pub model_alias: Option<String>,
    pub model_ftype: Option<String>,
    pub model_path: Option<String>,
    pub endpoint_slots: Option<bool>,
    pub endpoint_props: Option<bool>,
    pub endpoint_metrics: Option<bool>,
    pub build_info: Option<String>,
    pub is_sleeping: Option<bool>,
    pub default_generation_settings: Option<RawDefaultSettings>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct RawDefaultSettings {
    pub n_ctx: Option<u64>,
}

/// Parse a `/props` body. Unknown fields are ignored.
///
/// The top-level value must be a JSON object: serde would otherwise accept
/// other shapes (e.g. `[]`) and fall back to `RawProps::default()`, which
/// would make a malformed response look like a valid model with no fields.
pub fn parse_props(body: &str) -> Result<RawProps, String> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("invalid JSON: {e}"))?;
    match value {
        serde_json::Value::Object(_) => {
            serde_json::from_value(value).map_err(|e| format!("invalid JSON: {e}"))
        }
        _ => Err("expected a JSON object".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_body_is_an_error() {
        // A JSON array is not a props object: it must be rejected, otherwise
        // a 200 "[]" would look like a valid (empty) model.
        let err = parse_props("[]").expect_err("an array body is not a props object");
        assert_eq!(err, "expected a JSON object");
        // A scalar is rejected the same way.
        assert!(parse_props("42").is_err());
        assert!(parse_props("null").is_err());
    }

    #[test]
    fn parses_current_props_shape() {
        let body = r#"{
            "default_generation_settings": {"params": {"n_predict": -1}, "n_ctx": 4096},
            "total_slots": 2,
            "model_alias": "Qwen2.5-7B-Instruct",
            "model_ftype": "Q4_K_M",
            "model_path": "models/qwen.gguf",
            "modalities": {"vision": false, "video": false, "audio": false},
            "media_marker": "<__media_x__>",
            "endpoint_slots": true,
            "endpoint_props": true,
            "endpoint_metrics": true,
            "ui": true,
            "ui_settings": {},
            "chat_template": "some template",
            "chat_template_caps": {},
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "build_info": "b1234-abc123",
            "is_sleeping": false,
            "cors_proxy_enabled": false
        }"#;
        let props = parse_props(body).unwrap();
        assert_eq!(props.total_slots, Some(2));
        assert_eq!(props.model_alias.as_deref(), Some("Qwen2.5-7B-Instruct"));
        assert_eq!(props.build_info.as_deref(), Some("b1234-abc123"));
        assert_eq!(props.is_sleeping, Some(false));
        assert_eq!(props.default_generation_settings.as_ref().and_then(|s| s.n_ctx), Some(4096));
    }

    #[test]
    fn missing_fields_become_none() {
        let props = parse_props(r#"{"total_slots": 1}"#).unwrap();
        assert_eq!(props.total_slots, Some(1));
        assert_eq!(props.model_alias, None);
        assert_eq!(props.is_sleeping, None);
        assert_eq!(props.build_info, None);
    }

    #[test]
    fn null_is_sleeping_is_none() {
        let props = parse_props(r#"{"is_sleeping": null}"#).unwrap();
        assert_eq!(props.is_sleeping, None);
    }

    #[test]
    fn invalid_json_is_an_error() {
        assert!(parse_props("{nope").is_err());
    }

    #[test]
    fn fixture_props_parses() {
        let body = include_str!("../../../fixtures/props.json");
        let props = parse_props(body).unwrap();
        assert_eq!(props.total_slots, Some(2));
        assert_eq!(props.model_alias.as_deref(), Some("qwen3.8-27b"));
        assert_eq!(props.model_path.as_deref(), Some("models/Qwen3.8-27B-Q4_K_M.gguf"));
        assert_eq!(props.build_info.as_deref(), Some("b10488-9d77fa172"));
        assert_eq!(props.is_sleeping, Some(false));
        assert_eq!(props.default_generation_settings.as_ref().and_then(|s| s.n_ctx), Some(24576));
    }

    #[test]
    fn fixture_props_does_not_retain_template() {
        let body = include_str!("../../../fixtures/props.json");
        let props = parse_props(body).unwrap();
        let raw = serde_json::to_string(&props).unwrap();
        assert!(!raw.contains("template body never retained"));
    }

    #[test]
    fn template_text_is_not_retained() {
        let body = r#"{"model_alias": "m", "chat_template": "SECRET TEMPLATE"}"#;
        let props = parse_props(body).unwrap();
        let raw = serde_json::to_string(&props).unwrap();
        assert!(!raw.contains("SECRET"));
    }
}
