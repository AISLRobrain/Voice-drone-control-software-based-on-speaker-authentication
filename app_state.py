# app_state.py
import threading
from collections import deque
from dataclasses import dataclass, field
import queue
@dataclass
class AppState:
    
    session_ready: threading.Event = field(default_factory=threading.Event)
    shutdown_event: threading.Event = field(default_factory=threading.Event)
    sync_trigger_event: threading.Event = field(default_factory=threading.Event)

    ws_send_lock: threading.Lock = field(default_factory=threading.Lock)
    speech_time_lock: threading.Lock = field(default_factory=threading.Lock)

    verification_result_queue = queue.Queue()

    gesture_queue: deque = field(default_factory=deque)
    speech_start_time: float = 0.0
    speech_end_time: float = 0.0
    
    is_recording: bool = False
    is_speaker_verified: bool = False

    verification_status: str = "IDLE"  # "IDLE", "IN_PROGRESS", "FINISHED"