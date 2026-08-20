//! System-wide CPU/RAM and the monitored process.

use serde::{Deserialize, Serialize};

/// System-wide resource usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemSnapshot {
    pub cpu_usage_percent: Option<f64>,
    pub ram_used_bytes: Option<u64>,
    pub ram_total_bytes: Option<u64>,
    /// Candidate process data. `Some` only for a single local name match
    /// (`SingleLocalCandidate`) or a `Verified` association; never for
    /// remote endpoints or ambiguous (multiple-candidate) matches.
    pub process: Option<ProcessSnapshot>,
    /// How many processes on this host match the llama-server name
    /// (0 = none found). `None` when the process list could not be read or
    /// is not applicable (remote endpoint).
    #[serde(default)]
    pub process_match_count: Option<u32>,
    /// How strongly a local process can be tied to the configured endpoint.
    /// Name matching alone never proves an association.
    #[serde(default)]
    pub association: ProcessAssociation,
}

impl SystemSnapshot {
    pub fn ram_utilization(&self) -> Option<f64> {
        let used = self.ram_used_bytes?;
        let total = self.ram_total_bytes?;
        if total == 0 {
            return None;
        }
        Some(used as f64 / total as f64)
    }
}

/// A local process whose name matches the llama-server executable.
///
/// This is a *candidate*: nothing about it proves that it serves the
/// configured endpoint. The snapshot-level `association` field records how
/// much is actually known, so the UI never presents a wrong process as a
/// fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessSnapshot {
    pub pid: u32,
    pub name: String,
    pub cpu_usage_percent: Option<f64>,
    pub memory_bytes: Option<u64>,
    /// Seconds the process has been running, if known.
    pub uptime_secs: Option<u64>,
}

/// How strongly a local process can be tied to the configured endpoint.
///
/// A name match alone never proves that a process serves the endpoint, so
/// `Verified` is only ever reported when the association was technically
/// proven (endpoint-to-PID mapping). The Windows collector does not
/// currently implement that mapping, so it never reports `Verified`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessAssociation {
    /// The endpoint-to-PID association was technically verified.
    Verified,
    /// Exactly one local name match; relationship to the endpoint unproven.
    SingleLocalCandidate,
    /// Several local name matches; cannot say which one serves the endpoint.
    MultipleLocalCandidates,
    /// No local name match (HTTP monitoring continues).
    #[default]
    NoneFound,
    /// The endpoint host is not this machine; local processes are not
    /// matched or presented.
    RemoteEndpoint,
    /// The local process list could not be read.
    Unavailable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_utilization_none_when_missing() {
        let s = SystemSnapshot {
            cpu_usage_percent: None,
            ram_used_bytes: None,
            ram_total_bytes: Some(1000),
            process: None,
            process_match_count: None,
            association: ProcessAssociation::NoneFound,
        };
        assert!(s.ram_utilization().is_none());
    }

    #[test]
    fn association_default_is_none_found() {
        assert_eq!(ProcessAssociation::default(), ProcessAssociation::NoneFound);
    }

    #[test]
    fn association_serde_roundtrip() {
        let raw = serde_json::to_string(&ProcessAssociation::SingleLocalCandidate).unwrap();
        assert_eq!(raw, "\"single_local_candidate\"");
        let back: ProcessAssociation = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, ProcessAssociation::SingleLocalCandidate);
    }
}
