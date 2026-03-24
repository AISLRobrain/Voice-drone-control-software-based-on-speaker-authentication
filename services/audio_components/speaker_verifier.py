import os
import tempfile
from typing import Optional
from datetime import datetime  # 로그 시간 찍기

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import SpeakerRecognition


class SpeakerVerifier:
    """
    SpeechBrain의 verify_files API를 사용하여 화자 인증을 처리하는 클래스.
    - 입력 chunk: torch.Tensor(float32, 1D, [-1,1], 16kHz) 가정
    - VADHandler를 이용해 발화 구간만 버퍼링 후, 임시 wav로 저장하여 verify_files 수행
    """

    def __init__(self, config, state, vad_handler):
        self.config = config
        self.state = state
        self.device = config.DEVICE
        self.vad_handler = vad_handler

        # txt 로그 파일 경로 (없으면 여기에서 자동으로 생성됨)
        self.score_log_path = getattr(
            config,
            "SPEAKER_SCORE_LOG_PATH",
            "speaker_scores3.txt"
        )

        print("[SpeakerVerifier] 화자 인증 모델(verify_files) 로드 중...")
        try:
            self.verification = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device},
            )
        except Exception as e:
            raise RuntimeError(f"SpeechBrain 모델 로드 실패: {e}")

        # 최소 1개 레퍼런스는 있다고 가정
        ref = self.config.AUTHORIZED_SPEAKER_WAV_PATH
        if not os.path.exists(ref):
            raise FileNotFoundError(
                f"등록된 목소리 파일을 찾을 수 없습니다: {ref}\n"
                "경로 또는 파일 유효성을 확인하세요."
            )

        print("[SpeakerVerifier] 화자 인증 모델 로드 완료.")
        self.reset()

    def process_chunk(self, chunk: torch.Tensor) -> bool:
        """
        오디오 청크를 처리하여 화자 인증을 시도하고, 성공 여부를 반환합니다.
        """
        if not isinstance(chunk, torch.Tensor):
            raise TypeError("입력 chunk는 torch.Tensor여야 합니다.")

        is_speech = self.vad_handler.is_speech(chunk)

        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0

            # 발화 시작 감지
            if not self.is_speaking and self.speech_frames >= self.config.MIN_SPEECH_FRAMES:
                self.is_speaking = True
                print("[SpeakerVerifier] 발화 감지. 목소리 수집 시작...")

            # 수집 중이면 버퍼에 담기
            if self.is_speaking:
                i16 = (
                    torch.clamp(chunk, -1.0, 1.0)
                    .detach()
                    .cpu()
                    .contiguous()
                    .numpy()
                )
                i16 = (i16 * 32767.0).astype(np.int16)
                self.audio_buffer.extend(i16.tobytes())

        else:
            # 침묵 구간
            self.silence_frames += 1
            self.speech_frames = 0

            # 말하다가 충분히 조용해졌으면 인증 시도
            if self.is_speaking and self.silence_frames >= self.config.MIN_SILENCE_FRAMES:
                print("[SpeakerVerifier] 침묵 감지. 화자 인증 시도...")
                return self._verify()

        return False

    def _verify(self) -> bool:
        """
        버퍼에 쌓인 오디오로 임시 wav 파일을 생성하여 화자 인증을 수행합니다.
        """
        full_audio_bytes = bytes(self.audio_buffer)
        duration_s = len(full_audio_bytes) / (self.config.RATE_VAD * 2)
        self.audio_buffer.clear()

        # 다음 발화를 위해 상태 리셋
        self._reset_counters(clear_vad=True)

        # 최소 길이보다 짧으면 실패 처리
        min_bytes_to_verify = self.config.RATE_VAD * 2 * 0.5  # 예: 최소 0.5초
        if len(full_audio_bytes) < min_bytes_to_verify:
            print(f"  -> ❌ 인증 실패! (음성 데이터 부족: {len(full_audio_bytes)} bytes)")
            # 짧아서 실패한 것도 로그에 남기고 싶으면 여기서 기록
            self._append_score_log(
                success=False,
                scores=(None, None, None, None),
                reason="short_audio"
            )
            return False

        tmp_wav_path: Optional[str] = None
        try:
            # 임시 wav 생성
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_wav_path = tmp_file.name

            waveform_np_float = np.frombuffer(full_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sf.write(tmp_wav_path, waveform_np_float, self.config.RATE_VAD)

            # 4개의 등록 화자와 비교 (네 원래 코드 유지)
            score_tensor, prediction_tensor = self.verification.verify_files(
                self.config.AUTHORIZED_SPEAKER_WAV_PATH,
                tmp_wav_path
            )
            score_tensor_2, prediction_tensor_2 = self.verification.verify_files(
                self.config.AUTHORIZED_SPEAKER_WAV_PATH_2,
                tmp_wav_path
            )
            score_tensor_3, prediction_tensor_3 = self.verification.verify_files(
                self.config.AUTHORIZED_SPEAKER_WAV_PATH_3,
                tmp_wav_path
            )
            score_tensor_4, prediction_tensor_4 = self.verification.verify_files(
                self.config.AUTHORIZED_SPEAKER_WAV_PATH_4,
                tmp_wav_path
            )

            score = float(score_tensor.item())
            score_2 = float(score_tensor_2.item())
            score_3 = float(score_tensor_3.item())
            score_4 = float(score_tensor_4.item())

            print(
                f"  -> 유사도 점수: {score:.3f},{score_2:.3f},{score_3:.3f},{score_4:.3f} "
                f"(임계: {self.config.VERIFICATION_THRESHOLD:.3f})"
            )

            success = (
                score >= self.config.VERIFICATION_THRESHOLD
                or score_2 >= self.config.VERIFICATION_THRESHOLD
                or score_3 >= self.config.VERIFICATION_THRESHOLD
                or score_4 >= self.config.VERIFICATION_THRESHOLD
            )

            # 여기서 txt로 기록
            self._append_score_log(
                success=success,
                scores=(score, score_2, score_3, score_4),
                reason=None
            )

            if success:
                print("  -> ✅ 인증 성공!")
                return True
            else:
                print("  -> ❌ 인증 실패!")
                return False

        except Exception as e:
            print(f"  -> ⚠️ 인증 중 오류 발생: {e}")
            # 오류도 남겨두면 디버깅 좋음
            self._append_score_log(
                success=False,
                scores=(None, None, None, None),
                reason=f"exception:{e}"
            )
            return False
        finally:
            if tmp_wav_path and os.path.exists(tmp_wav_path):
                try:
                    os.remove(tmp_wav_path)
                except Exception:
                    pass

    def _append_score_log(self, *, success: bool, scores, reason: Optional[str]):
        """
        점수들을 사람이 읽기 쉬운 txt 형식으로 기록.
        예)
        [2025-11-12 15:30:01] success=1 scores=0.82,0.77,0.65,0.71 reason=
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        s1, s2, s3, s4 = scores
        line = (
            f"[{ts}] success={int(success)} "
            f"scores={s1},{s2},{s3},{s4} "
            f"reason={reason if reason else ''}\n"
        )

        # speaker_scores.txt는 append 모드로 열면 없을 때 자동 생성됨
        # 단, 상위 폴더가 없으면 에러이므로 파일명만 두는 게 안전
        try:
            with open(self.score_log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            print(f"[SpeakerVerifier] 점수 로그 기록 실패: {e}")

    def is_processing(self) -> bool:
        return self.is_speaking

    def reset(self):
        """인증 상태 및 버퍼를 초기화합니다."""
        self.audio_buffer = bytearray()
        self._reset_counters(clear_vad=True)

    def _reset_counters(self, *, clear_vad: bool):
        """내부 카운터만 초기화."""
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        if clear_vad:
            self.vad_handler.reset()

    # 외부에서 "지금까지 모인 걸로 일단 인증해봐" 하고 싶을 때
    def finalize_verification(self) -> bool:
        print("[SpeakerVerifier] 외부 신호에 의한 최종 인증 시도...")
        return self._verify()
