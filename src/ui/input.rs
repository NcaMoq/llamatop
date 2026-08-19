//! Keyboard input: raw key-to-action mapping and the input reader thread.
//!
//! The mapping lives here (not in the rendering code) so key handling is
//! testable without a terminal. The reader runs on a dedicated thread:
//! crossterm's event API is blocking, and the input must keep flowing even
//! while a slow HTTP fetch is in flight in the collector.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{spawn, JoinHandle};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use tokio::sync::mpsc::UnboundedSender;

use crate::app::event::{AppEvent, InputAction};

/// Map a raw key event to an application action, if any.
pub fn key_to_action(code: KeyCode, modifiers: KeyModifiers) -> Option<InputAction> {
    match code {
        KeyCode::Char('q') | KeyCode::Char('Q') => Some(InputAction::Quit),
        KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => Some(InputAction::Quit),
        KeyCode::Char('r') => Some(InputAction::Reconnect),
        KeyCode::Char('p') => Some(InputAction::TogglePause),
        KeyCode::Char('?') => Some(InputAction::ToggleHelp),
        KeyCode::Esc => Some(InputAction::CloseModal),
        KeyCode::Tab if modifiers.contains(KeyModifiers::SHIFT) => Some(InputAction::FocusPrev),
        KeyCode::Tab => Some(InputAction::FocusNext),
        KeyCode::Up | KeyCode::Char('k') => Some(InputAction::SlotUp),
        KeyCode::Down | KeyCode::Char('j') => Some(InputAction::SlotDown),
        KeyCode::Enter => Some(InputAction::ToggleSlotDetail),
        KeyCode::Char('l') => Some(InputAction::ToggleEvents),
        KeyCode::Char('c') => Some(InputAction::ClearHistory),
        _ => None,
    }
}

/// The input reader thread: maps keys and resizes into `AppEvent`s.
pub struct InputReader {
    handle: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
}

impl InputReader {
    /// Start the reader. It exits when `stop` is called or the event
    /// receiver is dropped.
    pub fn start(tx: UnboundedSender<AppEvent>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let handle = spawn(move || {
            loop {
                if stop_clone.load(Ordering::Relaxed) {
                    break;
                }
                // Bounded poll so the stop flag is checked regularly; the
                // reader thread is never the CPU hotspot.
                let pending = event::poll(Duration::from_millis(50));
                match pending {
                    Ok(true) => match event::read() {
                        Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                            if let Some(action) = key_to_action(key.code, key.modifiers) {
                                if tx.send(AppEvent::Input(action)).is_err() {
                                    break; // receiver gone
                                }
                            }
                        }
                        Ok(Event::Resize(width, height)) => {
                            if tx.send(AppEvent::Resize(width, height)).is_err() {
                                break;
                            }
                        }
                        Ok(_) => {}
                        Err(_) => {}
                    },
                    Ok(false) => {}
                    Err(_) => {}
                }
            }
        });
        Self { handle: Some(handle), stop }
    }

    /// Signal the thread to stop and join it.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for InputReader {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> Option<InputAction> {
        key_to_action(code, KeyModifiers::NONE)
    }

    #[test]
    fn q_and_ctrl_c_quit() {
        assert_eq!(key(KeyCode::Char('q')), Some(InputAction::Quit));
        assert_eq!(key(KeyCode::Char('Q')), Some(InputAction::Quit));
        assert_eq!(
            key_to_action(KeyCode::Char('c'), KeyModifiers::CONTROL),
            Some(InputAction::Quit)
        );
        // Plain c clears history, does not quit.
        assert_eq!(key(KeyCode::Char('c')), Some(InputAction::ClearHistory));
    }

    #[test]
    fn reconnect_pause_help_close_modal() {
        assert_eq!(key(KeyCode::Char('r')), Some(InputAction::Reconnect));
        assert_eq!(key(KeyCode::Char('p')), Some(InputAction::TogglePause));
        assert_eq!(key(KeyCode::Char('?')), Some(InputAction::ToggleHelp));
        assert_eq!(key(KeyCode::Esc), Some(InputAction::CloseModal));
    }

    #[test]
    fn tab_and_shift_tab_focus() {
        assert_eq!(key(KeyCode::Tab), Some(InputAction::FocusNext));
        assert_eq!(key_to_action(KeyCode::Tab, KeyModifiers::SHIFT), Some(InputAction::FocusPrev));
    }

    #[test]
    fn arrow_and_vi_keys_move_selection() {
        assert_eq!(key(KeyCode::Up), Some(InputAction::SlotUp));
        assert_eq!(key(KeyCode::Char('k')), Some(InputAction::SlotUp));
        assert_eq!(key(KeyCode::Down), Some(InputAction::SlotDown));
        assert_eq!(key(KeyCode::Char('j')), Some(InputAction::SlotDown));
    }

    #[test]
    fn enter_l_c_actions() {
        assert_eq!(key(KeyCode::Enter), Some(InputAction::ToggleSlotDetail));
        assert_eq!(key(KeyCode::Char('l')), Some(InputAction::ToggleEvents));
    }

    #[test]
    fn unrelated_keys_are_ignored() {
        assert_eq!(key(KeyCode::Char('x')), None);
        assert_eq!(key(KeyCode::F(9)), None);
    }
}
