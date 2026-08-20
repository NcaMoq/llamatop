//! System-wide CPU/RAM and the monitored process.

use serde::{Deserialize, Serialize};

/// System-wide resource usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemSnapshot {
    pub cpu_usage_percent: Option<f64>,
    pub ram_used_bytes: Option<u64>,
    pub ram_total_bytes: Option<u64>,
    /// `Some` only when exactly one matching llama-server process exists:
    /// with several candidates the endpoint association cannot be confirmed,
    /// so no single process is presented as the server.
    pub process: Option<ProcessSnapshot>,
    /// How many processes on this host match the llama-server name
    /// (0 = none found). `None` when the process list could not be read.
    #[serde(default)]
    pub process_match_count: Option<u32>,
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

/// The llama-server process, when it can be identified.
///
/// `identity` records how confident we are about which process is the server,
/// so the UI never presents a wrong process as a fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessSnapshot {
    pub pid: u32,
    pub name: String,
    pub cpu_usage_percent: Option<f64>,
    pub memory_bytes: Option<u64>,
    /// Seconds the process has been running, if known.
    pub uptime_secs: Option<u64>,
    pub identity: ProcessIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessIdentity {
    /// Exactly one matching process found.
    Exact,
    /// Several candidates; the endpoint association could not be confirmed.
    MultipleCandidates,
    /// No matching process found (HTTP monitoring continues).
    NoneFound,
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
        };
        assert!(s.ram_utilization().is_none());
    }
}
