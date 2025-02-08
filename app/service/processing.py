import ffmpeg
import torchaudio
import torchaudio.transforms as transforms
import soundfile as sf
import torch
import os
from pathlib import Path


class AudioProcessor:
    """Класс для обработки аудио и видео перед транскрипцией."""

    @staticmethod
    def extract_audio(video_path: str, output_audio_path: str, sample_rate: int = 16000):
        """
        Извлекает аудио из видеофайла и конвертирует его в WAV с 16 kHz.

        Args:
            video_path (str): Путь к видеофайлу.
            output_audio_path (str): Путь для сохранения извлеченного аудио.
            sample_rate (int, optional): Целевая частота дискретизации. По умолчанию 16 kHz.

        Returns:
            str: Путь к обработанному аудиофайлу.
        """
        print(f"🎥 Извлекаем аудио из видео: {video_path}")

        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_audio_path, format="wav", ar=sample_rate, ac=1)
                .run(overwrite_output=True, quiet=True)
            )
            print(f"✅ Аудио извлечено: {output_audio_path}")
            return output_audio_path
        except Exception as e:
            raise RuntimeError(f"Ошибка при извлечении аудио: {e}")

    @staticmethod
    def load_audio_file(audio_path: str, target_sample_rate: int = 16000):
        """
        Загружает аудиофайл и конвертирует его в 16 kHz WAV при необходимости.

        Поддерживаемые форматы: WAV, MP3, FLAC, OGG, AAC, M4A.

        Args:
            audio_path (str): Путь к аудиофайлу.
            target_sample_rate (int): Целевая частота дискретизации.

        Returns:
            torch.Tensor: Аудиоданные в формате (1, N).
            int: Частота дискретизации (обычно 16000).
        """
        audio_path = Path(audio_path)

        # Если формат не WAV, конвертируем
        if audio_path.suffix.lower() not in [".wav"]:
            print(f"⚠ Конвертируем {audio_path.suffix} в WAV...")
            temp_wav_path = audio_path.with_suffix(".wav")

            try:
                (
                    ffmpeg
                    .input(str(audio_path))
                    .output(str(temp_wav_path), format="wav", ar=target_sample_rate, ac=1)
                    .run(overwrite_output=True, quiet=True)
                )
                audio_path = temp_wav_path
            except Exception as e:
                raise RuntimeError(f"Ошибка при конвертации {audio_path}: {e}")

        # Загружаем аудиофайл через torchaudio
        try:
            audio, sample_rate = torchaudio.load(str(audio_path))
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки аудио: {e}")

        # Если частота не совпадает с 16 kHz, ресемплируем
        if sample_rate != target_sample_rate:
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            audio = resampler(audio)

        # Преобразуем стерео в моно
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return audio.to(torch.float32), target_sample_rate

    @staticmethod
    def save_audio(audio_tensor: torch.Tensor, output_path: str, sample_rate: int = 16000):
        """
        Сохраняет аудиоданные в WAV-файл.

        Args:
            audio_tensor (torch.Tensor): Аудио в формате (1, N).
            output_path (str): Путь сохранения файла.
            sample_rate (int, optional): Частота дискретизации (по умолчанию 16 kHz).
        """
        output_path = Path(output_path)
        sf.write(output_path, audio_tensor.squeeze(0).numpy(), sample_rate)
        print(f"📁 Аудио сохранено: {output_path}")
