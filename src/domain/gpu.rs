//! GPU metrics (provider-agnostic; NVML is one provider).

use serde::{Deserialize, Serialize};

/// A snapshot of one GPU. Every field is optional: a provider may fail to
/// read some values without failing the whole snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuSnapshot {
    pub index: u32,
    pub uuid: Option<String>,
    pub name: Option<String>,
    pub utilization_percent: Option<u8>,
    pub memory_used_bytes: Option<u64>,
    pub memory_total_bytes: Option<u64>,
    pub temperature_celsius: Option<u32>,
    pub power_watts: Option<f64>,
    pub power_limit_watts: Option<f64>,
    pub graphics_clock_mhz: Option<u32>,
    pub memory_clock_mhz: Option<u32>,
}

impl GpuSnapshot {
    /// VRAM utilization as a fraction 0.0..=1.0, if both used and total are known.
    pub fn memory_utilization(&self) -> Option<f64> {
        let used = self.memory_used_bytes?;
        let total = self.memory_total_bytes?;
        if total == 0 {
            return None;
        }
        Some(used as f64 / total as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_utilization_requires_both_values() {
        let g = GpuSnapshot {
            index: 0,
            uuid: None,
            name: None,
            utilization_percent: None,
            memory_used_bytes: Some(1000),
            memory_total_bytes: None,
            temperature_celsius: None,
            power_watts: None,
            power_limit_watts: None,
            graphics_clock_mhz: None,
            memory_clock_mhz: None,
        };
        assert!(g.memory_utilization().is_none());

        let g = GpuSnapshot { memory_total_bytes: Some(2000), ..g };
        assert!((g.memory_utilization().unwrap() - 0.5).abs() < 1e-9);
    }
}
