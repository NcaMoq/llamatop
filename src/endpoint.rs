//! Shared endpoint redaction for every display, error, and Debug surface.
//!
//! The configured endpoint must never appear in raw form: userinfo
//! (username/password), the query string, and the fragment are stripped so
//! a credential-bearing URL can never leak through the TUI, CLI output,
//! error messages, Debug output, or the JSON snapshot.
//!
//! `Config::validate` already rejects credential-bearing URLs; this
//! redaction is the defense-in-depth path every *output* site goes through,
//! including errors built from values that bypass validation.

use url::Url;

/// Redact an endpoint string for display: drop userinfo, query, and
/// fragment.
///
/// When the string parses as a URL, the `url` crate's API is used. When it
/// does not (e.g. the value is itself the subject of a parse error), a
/// conservative manual strip is applied so even a malformed
/// credential-bearing string never surfaces as-is.
pub fn redact(raw: &str) -> String {
    if let Ok(mut url) = Url::parse(raw) {
        // Only touch userinfo when present: setting an empty username on a
        // URL that has none would insert a stray "@".
        if !url.username().is_empty() || url.password().is_some() {
            let _ = url.set_username("");
            let _ = url.set_password(None);
        }
        url.set_query(None);
        url.set_fragment(None);
        return url.as_str().to_string();
    }

    // Manual fallback for unparseable input: cut the query/fragment, then
    // drop userinfo ("scheme://user:pass@host" -> "scheme://host").
    let trimmed = raw.split(['?', '#']).next().unwrap_or("");
    if let (Some(at), Some(scheme_end)) = (trimmed.find('@'), trimmed.find("://")) {
        if at > scheme_end {
            return format!("{}{}", &trimmed[..scheme_end + 3], &trimmed[at + 1..]);
        }
    }
    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_strips_userinfo_query_and_fragment() {
        assert_eq!(
            redact("http://user:secret@example.com:8080/?token=t#frag"),
            "http://example.com:8080/"
        );
        assert_eq!(redact("http://user@example.com:8080"), "http://example.com:8080/");
        assert_eq!(redact("https://127.0.0.1:8080/?key=v"), "https://127.0.0.1:8080/");
    }

    #[test]
    fn redact_is_a_noop_for_clean_urls() {
        assert_eq!(redact("http://127.0.0.1:8080"), "http://127.0.0.1:8080/");
        assert_eq!(redact("http://127.0.0.1:8080/"), "http://127.0.0.1:8080/");
        assert_eq!(redact("https://llama.example.com:9091/"), "https://llama.example.com:9091/");
    }

    #[test]
    fn redact_handles_unparseable_input_without_panicking() {
        // Malformed but credential-bearing: still stripped, never echoed.
        assert_eq!(redact("http://user:pass@not a url"), "http://not a url");
        assert_eq!(redact("not a url"), "not a url");
        assert_eq!(redact(""), "");
        // The redacted output must never contain the secret.
        let out = redact("http://user:hunter2@x:8080/?a=b");
        assert!(!out.contains("hunter2"));
        assert!(!out.contains("user"));
    }
}
