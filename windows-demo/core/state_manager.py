"""
Application state manager for Blind Navigation Assistant.
Handles state transitions and coordinates between components.
"""
import threading
from typing import Callable, List, Optional
from enum import Enum

from utils.config import AppState
from utils.logger import get_logger


class StateManager:
    """
    Centralized state machine for the application.
    Thread-safe implementation for multi-threaded operation.
    """
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        AppState.IDLE: [AppState.SHOE_SETUP, AppState.STARTING, AppState.ERROR],
        AppState.SHOE_SETUP: [AppState.STARTING, AppState.IDLE, AppState.ERROR],
        AppState.STARTING: [AppState.ACTIVE, AppState.IDLE, AppState.ERROR],
        AppState.ACTIVE: [AppState.PROCESSING, AppState.WARNING, AppState.PAUSED, AppState.IDLE, AppState.ERROR],
        AppState.PROCESSING: [AppState.ACTIVE, AppState.WARNING, AppState.PAUSED, AppState.IDLE, AppState.ERROR],
        AppState.WARNING: [AppState.ACTIVE, AppState.PAUSED, AppState.IDLE, AppState.ERROR],
        AppState.PAUSED: [AppState.ACTIVE, AppState.IDLE, AppState.ERROR],
        AppState.ERROR: [AppState.IDLE, AppState.SHOE_SETUP, AppState.STARTING],
    }
    
    def __init__(self, initial_state: AppState = AppState.IDLE):
        """
        Initialize state manager.
        
        Args:
            initial_state: Starting state of the application
        """
        self.logger = get_logger("state")
        self._lock = threading.RLock()
        self._state = initial_state
        self._listeners: List[Callable[[AppState, AppState], None]] = []
        
        self.logger.info(f"StateManager initialized with state: {initial_state.value}")
    
    @property
    def state(self) -> AppState:
        """Get current application state."""
        with self._lock:
            return self._state
    
    def set_state(self, new_state: AppState, force: bool = False) -> bool:
        """
        Transition to a new state.
        
        Args:
            new_state: Target state
            force: If True, skip transition validation
            
        Returns:
            True if transition successful, False otherwise
        """
        with self._lock:
            old_state = self._state
            
            # Check if transition is valid
            if not force and new_state not in self.VALID_TRANSITIONS.get(old_state, []):
                self.logger.warning(
                    f"Invalid state transition: {old_state.value} -> {new_state.value}"
                )
                return False
            
            # Same state, no change needed
            if old_state == new_state:
                return True
            
            # Perform transition
            self._state = new_state
            self.logger.info(f"State transition: {old_state.value} -> {new_state.value}")
            
            # Notify listeners
            self._notify_listeners(old_state, new_state)
            
            return True
    
    def add_listener(self, callback: Callable[[AppState, AppState], None]):
        """
        Add a state change listener.
        
        Args:
            callback: Function to call on state change (old_state, new_state)
        """
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable[[AppState, AppState], None]):
        """
        Remove a state change listener.
        
        Args:
            callback: Previously registered callback to remove
        """
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)
    
    def _notify_listeners(self, old_state: AppState, new_state: AppState):
        """Notify all listeners of state change."""
        for listener in self._listeners:
            try:
                listener(old_state, new_state)
            except Exception as e:
                self.logger.error(f"Listener error: {e}")
    
    def is_active(self) -> bool:
        """Check if navigation is currently active."""
        with self._lock:
            return self._state in (AppState.ACTIVE, AppState.PROCESSING, AppState.WARNING)
    
    def is_error(self) -> bool:
        """Check if application is in error state."""
        with self._lock:
            return self._state == AppState.ERROR
    
    def can_start(self) -> bool:
        """Check if navigation can be started from current state."""
        with self._lock:
            return AppState.STARTING in self.VALID_TRANSITIONS.get(self._state, [])
    
    def is_shoe_setup(self) -> bool:
        """Check if in shoe setup state."""
        with self._lock:
            return self._state == AppState.SHOE_SETUP
    
    def reset(self):
        """Reset state machine to idle state."""
        self.set_state(AppState.IDLE, force=True)
        self.logger.info("State machine reset to IDLE")
