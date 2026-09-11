"""Self-play training tools for Dots Cordon."""

from .encoding import INPUT_CHANNELS, encode_state, legal_action_mask

__all__ = ["INPUT_CHANNELS", "encode_state", "legal_action_mask"]

