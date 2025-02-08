# 🗣️ Speech-to-Text API (Whisper)

**Speech-to-Text API** is a microservice based on **FastAPI** and **Whisper (OpenAI)** that transcribes audio and video into text.  
The service supports **99+ languages**, including **English, Russian, Chinese, Spanish, and many more**.

  The project is under development! Some functionality is still in the process of being written, and changes in the API and operating logic are possible.

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
git clone https://github.com/your-repo/speech-to-text.git
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

### **1️⃣ Upload an Audio File**
📌 **POST `/transcribe/`** (send an audio file and receive text)
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/transcribe/' \
  -F 'file=@audio.mp3'
```
🔹 **Server Response:**
```json
{
  "filename": "audio.mp3",
  "transcription": "This is an example transcription"
}
```

---

### **2️⃣ Transcription with Translation to English**
📌 **Use `task="translate"`**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/transcribe/' \
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
