//! Unicode/ASCII symbol set shared by CLI output and the TUI.
//!
//! States are never conveyed by color alone: every symbol is paired with a
//! text label at the point of rendering.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Symbols {
    ascii: bool,
}

impl Symbols {
    pub fn new(ascii: bool) -> Self {
        Self { ascii }
    }

    pub fn success(&self) -> &'static str {
        if self.ascii {
            "[OK]"
        } else {
            "✓"
        }
    }

    pub fn warning(&self) -> &'static str {
        if self.ascii {
            "[WARN]"
        } else {
            "▲"
        }
    }

    pub fn error(&self) -> &'static str {
        if self.ascii {
            "[ERR]"
        } else {
            "✘"
        }
    }

    pub fn active(&self) -> &'static str {
        if self.ascii {
            "[RUN]"
        } else {
            "●"
        }
    }

    pub fn idle(&self) -> &'static str {
        if self.ascii {
            "[IDLE]"
        } else {
            "○"
        }
    }

    /// A horizontal rule of the given width in the current symbol set.
    pub fn separator(&self, width: usize) -> String {
        let ch = if self.ascii { '-' } else { '─' };
        ch.to_string().repeat(width)
    }
}

impl Default for Symbols {
    fn default() -> Self {
        Self::new(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_symbols() {
        let s = Symbols::new(false);
        assert_eq!(s.success(), "✓");
        assert_eq!(s.warning(), "▲");
        assert_eq!(s.error(), "✘");
    }

    #[test]
    fn ascii_symbols() {
        let s = Symbols::new(true);
        assert_eq!(s.success(), "[OK]");
        assert_eq!(s.warning(), "[WARN]");
        assert_eq!(s.error(), "[ERR]");
        assert_eq!(s.active(), "[RUN]");
        assert_eq!(s.idle(), "[IDLE]");
    }
}
