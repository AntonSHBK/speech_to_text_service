import time
from typing import List, Optional, Union
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from app.models.base import BaseModel


class FastWhisperTranscriber(BaseModel):
    
    def __init__(
        self,
        model_name: str, 
        cache_dir: Path = 'cache_dir', 
        device: str = 'cpu', 
        token: str = None,
        compute_type: str = "default",
        cpu_threads: int = 4,
        num_workers: int = 8
        
    ):
        super().__init__(model_name, cache_dir, device)
        self.model = self.load_model(
            model_name, 
            use_auth_token=token, 
            compute_type=compute_type,
            cpu_threads=cpu_threads, 
            num_workers=num_workers
        )

    def load_model(
        self, 
        model_name: str,
        use_auth_token: str = None,
        compute_type: str = "default",
        cpu_threads: int = 4,
        num_workers: int = 8
    ) -> "WhisperModel":      
        model = WhisperModel(
            model_size_or_path=model_name,
            device=self.device,
            download_root=str(self.cache_dir),
            use_auth_token=use_auth_token,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers
        )
        self.logger.info(f"Модель {model_name} загружена.")
        return model
    
    def process(self, audio_path: str | Path) -> Path:
        return audio_path

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = "ru",
        task: str = "transcribe",
        beam_size: int = 3,
        chunk_length: int = 10,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.0,
        multilingual: bool = False,
        **kwargs
    ) -> dict:

        self.logger.info(f"Старт транскрипции: {audio_path}")
        
        audio_path = self.process(audio_path)
        
        start_time = time.perf_counter()

        segments, info = self.model.transcribe(
            audio=str(audio_path),
            language=language,
            task=task,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            beam_size=beam_size,
            chunk_length=chunk_length,
            multilingual=multilingual,
            **kwargs,
        )

        collected_segments: List[Segment] = []
        text_parts: List[str] = []
        
        last_logged = 0   
        
        for s in segments:
            collected_segments.append(s)
            text_parts.append(s.text.strip())

            progress = min(100.0, (s.end / info.duration) * 100)

            if int(progress) >= last_logged + 2:
                last_logged = int(progress)
                self.logger.info(f"Прогресс: {progress:.1f}%")
                
        end_time = time.perf_counter()

        processing_time = end_time - start_time
        audio_duration = info.duration
        speed_ratio = audio_duration / processing_time if processing_time > 0 else 0

        self.logger.info(
            f"Транскрипция завершена: {audio_path} | "
            f"Длительность аудио: {audio_duration:.2f} сек | "
            f"Время обработки: {processing_time:.2f} сек | "
            f"Скорость: x{speed_ratio:.2f} секунд видео в секунду"
        )

        full_text = " ".join(
            part.strip()
            for part in text_parts
                if part and part.strip()
        )

        return {
            "language": info.language,
            "duration": info.duration,
            "text": full_text,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in collected_segments
            ],
        }


# class WhisperTranscriber(BaseModel):
#     """Транскрибатор на базе openai/whisper (Transformers)."""

#     def __init__(
#         self,
#         model_name: str = "openai/whisper-small",
#         cache_dir: Path = "cache_dir",
#         device: str = "cpu",
#     ):
#         super().__init__(model_name, cache_dir, device)
#         self.model, self.processor = self.load_model(model_name)

#     def load_model(self, model_name: str):
#         self.processor = WhisperProcessor.from_pretrained(model_name, cache_dir=self.cache_dir)
#         self.model = WhisperForConditionalGeneration.from_pretrained(
#             model_name, cache_dir=self.cache_dir
#         ).to(self.device)
#         return self.model, self.processor


#     def _load_audio(self, audio_path: str) -> AudioSegment:
#         audio: AudioSegment = AudioSegment.from_file(audio_path)
#         audio = audio.set_channels(1)
#         audio = audio.set_frame_rate(16000)
#         return audio

#     def _split_audio(self, audio: AudioSegment, chunk_length_ms: int) -> List[AudioSegment]:
#         return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

#     def process(self, audio_path: str, chunk_length_ms: int = 30_000) -> List[io.BytesIO]:
#         audio = self._load_audio(audio_path)
#         chunks = self._split_audio(audio, chunk_length_ms)

#         buffers = []
#         for chunk in chunks:
#             buf = io.BytesIO()
#             chunk.export(buf, format="wav")
#             buf.seek(0)
#             buffers.append(buf)
#         return buffers


#     def _infer(self, audio_buffer: io.BytesIO, language: str, task: str, **gen_kwargs):
#         audio = AudioSegment.from_file(audio_buffer).get_array_of_samples()
#         audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0) / 32768.0

#         inputs = self.processor(audio.squeeze(0), sampling_rate=16000, return_tensors="pt")
#         input_features = inputs.input_features.to(self.device)

#         forced_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)

#         with torch.no_grad():
#             predicted_ids = self.model.generate(
#                 input_features,
#                 forced_decoder_ids=forced_ids,
#                 **gen_kwargs,
#             )

#         return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

#     def transcribe(
#         self,
#         audio_path: str,
#         chunk_length_ms: int = 30_000,
#         language: str = "ru",
#         task: str = "transcribe",
#         **gen_kwargs,
#     ) -> dict:

#         buffers = self.process(audio_path, chunk_length_ms)

#         full_text = []

#         for buf in buffers:
#             text = self._infer(buf, language, task, **gen_kwargs).strip()
#             if text:
#                 full_text.append(text)

#         return {
#             "language": language,
#             "duration": len(buffers) * (chunk_length_ms / 1000),
#             "text": " ".join(full_text),
#         }
