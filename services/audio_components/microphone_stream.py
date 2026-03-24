# services/audio_components/microphone_stream.py
from __future__ import annotations
import pyaudio
import threading
from collections import deque
from typing import Iterator, Optional

from ..interfaces import IAudioStream


class MicrophoneStream(IAudioStream):
    """
    PyAudio 콜백 모드 기반 마이크 입력.
    - 16-bit PCM, mono, RATE_MIC / CHUNK_MIC
    - 콜백 스레드에서 들어오는 오디오를 내부 큐(deque)에 적재
    - listen() 제너레이터가 큐에서 안전하게 소비
    """

    def __init__(
        self,
        config,
        *,
        device_index: Optional[int] = None,
        max_queue_chunks: int = 128,   # 큐 최대 청크 수 (백프레셔/메모리 보호)
        warn_on_drop: bool = True,
    ):
        self.config = config
        self._pa = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None

        self._device_index = device_index if device_index is not None else self.config.MIC_INDEX

        # 내부 큐 & 동기화
        self._queue = deque(maxlen=max_queue_chunks)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._running = True
        self._drop_warned = False
        self._warn_on_drop = warn_on_drop

        # 스트림 열기 (콜백 모드)
        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.RATE_MIC,
                input=True,
                frames_per_buffer=self.config.CHUNK_MIC,
                input_device_index=self._device_index,
                stream_callback=self._callback,  # 콜백 등록!
                start=True,                      # 즉시 시작
            )
        except Exception as e:
            print(f"[🎙️] 마이크 열기 실패: {e}")
            self._print_input_devices()
            self._cleanup_pa()
            raise

        try:
            info = self._pa.get_device_info_by_index(self._device_index)
            latency_ms = (info.get("defaultLowInputLatency", 0.0)) * 1000
            print(
                f"[🎙️] 마이크(콜백) 열림: device='{info.get('name','?')}', "
                f"rate={self.config.RATE_MIC}, chunk={self.config.CHUNK_MIC} frames, "
                f"low-latency≈{latency_ms:.1f} ms"
            )
        except Exception:
            print("[🎙️] 마이크 스트림이 열렸습니다. (장치 정보 조회 실패)")

    # ---------- PyAudio 콜백 ----------
    def _callback(self, in_data, frame_count, time_info, status_flags):
        # PyAudio 내부 스레드 문맥: 최소한의 작업만 하고 빨리 반환
        with self._cv:
            if len(self._queue) == self._queue.maxlen:
                # 꽉 찬 경우 가장 오래된 청크 drop (백프레셔)
                self._queue.popleft()
                if self._warn_on_drop and not self._drop_warned:
                    self._drop_warned = True
                    print("[🎙️] 경고: 입력 큐 가득 참 → 오래된 청크 drop (한 번만 경고)")
            self._queue.append(in_data)
            self._cv.notify()  # 소비자 깨우기
        return (None, pyaudio.paContinue)

    # ---------- 소비자(메인) ----------
    def listen(self) -> Iterator[bytes]:
        """
        큐에서 오디오 청크를 꺼내 바이트로 내보냄.
        외부 루프가 종료될 때까지(blocking) 동작.
        """
        while self._running and self._stream and self._stream.is_active():
            with self._cv:
                while self._running and not self._queue:
                    self._cv.wait(timeout=0.2)  # 깨우기 대기 (종료/타임아웃 체크)
                if not self._running:
                    break
                if self._queue:
                    chunk = self._queue.popleft()
                else:
                    continue
            yield chunk  # 락 밖에서 전달

    # ---------- 종료/정리 ----------
    def close(self):
        """자원 정리 (idempotent)"""
        self._running = False
        with self._cv:
            self._cv.notify_all()
        try:
            if self._stream:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
        finally:
            self._stream = None
            self._cleanup_pa()
            print("[🎙️] 마이크 스트림이 닫혔습니다.")

    def _cleanup_pa(self):
        try:
            self._pa.terminate()
        except Exception:
            pass

    # 컨텍스트 매니저 지원
    def __enter__(self) -> "MicrophoneStream":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------- 유틸 ----------
    def _print_input_devices(self):
        try:
            count = self._pa.get_device_count()
        except Exception:
            print("[🎙️] 오디오 장치 목록을 가져오지 못했습니다.")
            return
        print("[🎙️] 사용 가능한 입력 장치 목록:")
        for i in range(count):
            try:
                info = self._pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    print(
                        f"  - index={i:2d} | name={info.get('name','?')} | "
                        f"defaultSR={int(info.get('defaultSampleRate',0))}"
                    )
            except Exception:
                pass
