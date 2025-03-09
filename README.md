# 🗣️ Speech-to-Text API (Whisper)

**Speech-to-Text API** is a microservice based on **FastAPI** and **Whisper (OpenAI)** that transcribes audio and video into text.  
The service supports **99+ languages**, including **English, Russian, Chinese, Spanish, and many more**.

## 🚀 Features
- 🎤 **Audio transcription** (WAV, MP3, FLAC, AAC, OGG, M4A, etc.)
- 🎥 **Extract audio from video** (MP4, MKV, AVI)
- 🌍 **Supports 99+ languages** (see the table below)
- 🔄 **Optional translation into English**
- 🏗 **Asynchronous API for integration**
- 🐳 **Docker-ready deployment**

---

## 🌍 Supported Languages

| #  | Language        | Code | #  | Language       | Code |
|----|---------------|------|----|--------------|------|
| 1  | English       | `en` | 6  | Spanish       | `es` |
| 2  | Russian       | `ru` | 7  | Portuguese    | `pt` |
| 3  | Chinese       | `zh` | 8  | Arabic        | `ar` |
| 4  | French        | `fr` | 9  | Japanese      | `ja` |
| 5  | German        | `de` | 10 | Hindi         | `hi` |

Whisper supports **99+ languages**. The full list is available in the [OpenAI repository](https://github.com/openai/whisper).

---

## 🛠️ Installation & Setup

### **1️⃣ Run Locally (without Docker)**
> **Requirements**: Python 3.9+, FFmpeg

```bash
git clone https://github.com/AntonSHBK/speech_to_text_service
cd speech-to-text

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Now the API is available at **`http://127.0.0.1:8000/docs`** 📜

---

### **2️⃣ Run with Docker**
> **Requirements**: Docker & Docker Compose

```bash
# Build the container
docker-compose build

# Start the service
docker-compose up -d
```
Now the API is running in a container and available at **`http://localhost:8000/docs`**

---

## 🎯 How to Use the API?

### **📌 1️⃣ Upload an Audio File**
📌 **POST `/transcribe/`** (Send an audio file and receive text)
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/transcribe/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@audio.mp3'
```
🔹 **Server Response:**
```json
{
  "filename": "audio.mp3",
  "transcription": "This is an example transcription"
}
```
✅ **Supported formats:** `wav, mp3, flac, ogg, aac, m4a, wma, aiff`  

---

### **📌 2️⃣ Transcription with Translation to English**
📌 **Use `task="translate"`**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/transcribe/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@audio.mp3' \
  -F 'task=translate'
```
🔹 **Server Response:**
```json
{
  "filename": "audio.mp3",
  "transcription": "This is an example transcription in English"
}
```

---

### **📌 3️⃣ Customize Transcription with Parameters**
You can control how the transcription is generated using various parameters.

#### 🔹 **Available parameters:**
| Parameter         | Description                                   | Default  |
|------------------|---------------------------------------------|----------|
| `language`       | Language code (`ru`, `en`, `fr`, etc.)      | `"ru"`   |
| `task`           | `"transcribe"` (default) or `"translate"`   | `"transcribe"` |
| `temperature`    | Controls randomness (0.0 = deterministic)  | `0.1`    |
| `max_new_tokens` | Max length of transcription output         | `100`    |
| `repetition_penalty` | Penalizes repeated words               | `1.2`    |
| `num_beams`      | Beam search size (higher = better quality) | `1`      |
| `do_sample`      | Enables sampling (more variability)        | `false`  |
| `top_k`          | Selects next word from `k` best options    | `50`     |
| `top_p`          | Nucleus sampling threshold                 | `0.95`   |

#### 📌 **Example: More accurate transcription (beam search + repetition penalty)**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/transcribe/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@audio.mp3' \
  -F 'language=en' \
  -F 'num_beams=5' \
  -F 'repetition_penalty=1.1'
```

---

## 🎯 Use Cases

✅ **1. Transcribing interviews & lectures** — Automatic speech-to-text for meetings and podcasts  
✅ **2. Voice notes & subtitles** — Convert speech into subtitles for videos and recordings  
✅ **3. Call transcription** — Process phone conversations and customer support calls  
✅ **4. Voice commands & chatbots** — Integrate into voice assistants and AI chatbots  

---

## 🛠 Support & Contributions
If you have ideas for improvement or found a bug, create an **Issue** or **Pull Request** in the [project repository](https://github.com/AntonSHBK/speech_to_text_service).

**👨‍💻 Author:** [Anton Pisarenko](https://github.com/AntonSHBK)  
**🔗 License:** Apache License 2.0  

🚀 **Start using the Speech-to-Text API today!** 🎙️
