import os
import ffmpeg
import torchaudio
import torchaudio.transforms as transforms
import soundfile as sf
from pathlib import Path
import torch


def convert_to_wav(input_path: str, output_path: str, target_sample_rate: int = 16000):
    """
    Конвертирует аудиофайл в WAV с частотой дискретизации 16 kHz (если требуется).

    Args:
        input_path (str): Путь к входному файлу.
        output_path (str): Путь к выходному файлу WAV.
        target_sample_rate (int): Частота дискретизации (по умолчанию 16 kHz).

    Returns:
        str: Путь к выходному файлу WAV.
    """
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, format="wav", ar=target_sample_rate, ac=1)
            .run(overwrite_output=True, quiet=True)
        )
        return output_path
    except Exception as e:
        raise RuntimeError(f"Ошибка конвертации файла {input_path} → {output_path}: {e}")


def extract_audio_from_video(video_path: str, output_audio_path: str, target_sample_rate: int = 16000):
    """
    Извлекает аудиодорожку из видеофайла и сохраняет ее как WAV.

    Args:
        video_path (str): Путь к видеофайлу.
        output_audio_path (str): Путь для сохранения аудиофайла.
        target_sample_rate (int): Частота дискретизации.

    Returns:
        str: Путь к извлеченному аудиофайлу.
    """
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, format="wav", ar=target_sample_rate, ac=1)
            .run(overwrite_output=True, quiet=True)
        )
        return output_audio_path
    except Exception as e:
        raise RuntimeError(f"Ошибка извлечения аудио из {video_path}: {e}")


def load_audio(audio_path: str, target_sample_rate: int = 16000) -> torch.Tensor:
    """
    Загружает аудиофайл, ресемплирует (если требуется) и возвращает тензор.

    Args:
        audio_path (str): Путь к аудиофайлу.
        target_sample_rate (int): Целевая частота дискретизации.

    Returns:
        torch.Tensor: Аудиоданные в формате (1, N).
    """
    try:
        audio, sample_rate = torchaudio.load(audio_path)

        # Если частота дискретизации не соответствует 16 kHz, ресемплируем
        if sample_rate != target_sample_rate:
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            audio = resampler(audio)

        # Приводим стерео в моно (усредняем каналы)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return audio
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки аудио {audio_path}: {e}")


def save_audio(audio_tensor: torch.Tensor, sample_rate: int, output_path: str):
    """
    Сохраняет аудиотензор в файл WAV.

    Args:
        audio_tensor (torch.Tensor): Аудиоданные.
        sample_rate (int): Частота дискретизации.
        output_path (str): Путь для сохранения.
    """
    try:
        sf.write(output_path, audio_tensor.numpy().squeeze(), sample_rate)
    except Exception as e:
        raise RuntimeError(f"Ошибка сохранения аудиофайла {output_path}: {e}")
