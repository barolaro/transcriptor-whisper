import streamlit as st
import whisper
from docx import Document
import tempfile
import os
import subprocess
from pathlib import Path

st.set_page_config(page_title="Transcriptor de Audio", layout="centered")

st.title("📝 Transcriptor de Audio con Whisper")
st.write(
    "Sube un archivo de audio y genera su transcripción en texto y Word. "
    "Formatos compatibles: mp3, wav, m4a, mp4, aac."
)

@st.cache_resource
def load_model():
    return whisper.load_model("base")


def convertir_a_wav(input_path):
    output_path = input_path + "_convertido.wav"

    comando = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]

    result = subprocess.run(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise Exception("No se pudo convertir el audio. Verifica que ffmpeg esté instalado.")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise Exception("El archivo convertido quedó vacío.")

    return output_path


def obtener_duracion_audio(path):
    comando = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]

    result = subprocess.run(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        return float(result.stdout.strip())
    except Exception:
        return 0


uploaded_file = st.file_uploader(
    "📂 Sube tu archivo de audio",
    type=["mp3", "wav", "m4a", "mp4", "aac"]
)

if uploaded_file is not None:
    tmp_path = None
    wav_path = None
    word_file = None

    try:
        suffix = Path(uploaded_file.name).suffix.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        duracion = obtener_duracion_audio(tmp_path)

        if duracion < 1:
            st.warning("⚠️ El audio parece estar vacío o dura menos de 1 segundo.")
            st.stop()

        with st.spinner("🔄 Convirtiendo audio a formato compatible..."):
            wav_path = convertir_a_wav(tmp_path)

        with st.spinner("🔄 Cargando modelo Whisper..."):
            model = load_model()

        with st.spinner("🔄 Transcribiendo audio, espera un momento..."):
            result = model.transcribe(
                wav_path,
                language="es",
                fp16=False,
                verbose=False,
                condition_on_previous_text=False
            )

        transcription = result.get("text", "").strip()

        if not transcription:
            st.warning("⚠️ No se detectó texto. Verifica que el audio tenga voz clara.")
        else:
            st.success("✅ Transcripción completa")

            st.subheader("📄 Texto transcrito:")
            st.write(transcription)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_docx:
                word_file = tmp_docx.name

            doc = Document()
            doc.add_heading("Transcripción de Audio", 0)
            doc.add_paragraph(transcription)
            doc.save(word_file)

            with open(word_file, "rb") as f:
                st.download_button(
                    label="📥 Descargar Word",
                    data=f.read(),
                    file_name="transcripcion.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    except Exception as e:
        st.error(f"❌ Error al procesar el audio: {e}")

    finally:
        for path in [tmp_path, wav_path, word_file]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
