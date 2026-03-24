# services/audio_components/wake_word_detector.py

import pvporcupine
import numpy as np
import torch

class WakeWordDetector:
    """호출어(Wake Word) 감지를 처리하는 클래스"""

    def __init__(self, config):
        self.config = config
        print("[WakeWordDetector] Porcupine 엔진 로드 중...")
        try:
            self.porcupine = pvporcupine.create(
                access_key=self.config.PICOVOICE_ACCESS_KEY,
                keyword_paths=self.config.KEYWORD_PATHS,
                model_path=self.config.PORCUPINE_MODEL_PATH
            )
        except pvporcupine.PorcupineError as e:
            print(f"Porcupine 엔진 초기화 오류: {e}")
            print("액세스 키나 모델 경로가 올바른지 확인하세요.")
            exit()

        self.frame_length = self.porcupine.frame_length
        self.sample_rate = self.porcupine.sample_rate
        self.buffer = np.zeros(0, dtype=np.int16)
        print("[WakeWordDetector] Porcupine 엔진 로드 완료.")

        if self.sample_rate != self.config.RATE_VAD:
            print(f"경고: Porcupine 샘플레이트({self.sample_rate})와 VAD 레이트({self.config.RATE_VAD})가 다릅니다. "
                  f"AudioProcessor에서 Porcupine 레이트로 리샘플해 전달해야 합니다.")

    def reset(self):
        """상태 전환 시 내부 버퍼 초기화"""
        self.buffer = np.zeros(0, dtype=np.int16)

    def _to_int16_numpy(self, pcm_chunk):
        """torch.Tensor(float32/-1~1) 또는 np.ndarray(int16/float32)를 int16 numpy로 통일"""
        if isinstance(pcm_chunk, torch.Tensor):
            # 기대: float32 [-1, 1], 1-D
            t = torch.clamp(pcm_chunk, -1.0, 1.0)
            return (t.detach().cpu().contiguous().numpy() * 32767.0).astype(np.int16)

        if isinstance(pcm_chunk, np.ndarray):
            if pcm_chunk.dtype == np.int16:
                return pcm_chunk
            # float형이라면 [-1,1]로 가정하고 스케일링
            if np.issubdtype(pcm_chunk.dtype, np.floating):
                pcm = np.clip(pcm_chunk, -1.0, 1.0)
                return (pcm * 32767.0).astype(np.int16)

        raise TypeError("입력은 torch.Tensor(float32 [-1,1]) 또는 NumPy 배열(int16/float) 이어야 합니다.")

    def process(self, pcm_chunk) -> bool:
        """
        1-D 오디오 청크를 받아 호출어를 감지합니다.
        - 입력: torch.Tensor(float32, -1~1) 또는 np.ndarray(int16/float)
        - 레이트: self.sample_rate(대개 16kHz)와 같아야 함
        """
        pcm_i16 = self._to_int16_numpy(pcm_chunk)

        # 버퍼에 이어붙이기 (np.append 대신 concat로 묶음 처리)
        if self.buffer.size == 0:
            self.buffer = pcm_i16
        else:
            self.buffer = np.concatenate((self.buffer, pcm_i16))

        # frame_length 단위로 처리
        detected = False
        while self.buffer.size >= self.frame_length:
            frame = self.buffer[:self.frame_length]
            self.buffer = self.buffer[self.frame_length:]

            keyword_index = self.porcupine.process(frame)
            if keyword_index >= 0:
                print("WakeWordDetector: 호출어 감지!")
                self.reset()   # 감지 후 버퍼 클리어
                detected = True
                break

        return detected

    def release(self):
        """Porcupine 엔진 리소스를 해제합니다."""
        if getattr(self, "porcupine", None) is not None:
            self.porcupine.delete()
            print("[WakeWordDetector] Porcupine 엔진 리소스 해제 완료.")
