import streamlit as st
import whisper
from docx import Document
import tempfile
import os

st.set_page_config(page_title="Transcriptor de Audio", layout="centered")
st.title("📝 Transcriptor de Audio con Whisper")
st.write("Sube un archivo de audio y genera su transcripción en texto y Word. Formatos compatibles: mp3, wav, m4a, mp4, aac.")

# Cargar el modelo UNA sola vez y cachearlo para no recargarlo en cada interacción
@st.cache_resource
def load_model():
    return whisper.load_model("base")

uploaded_file = st.file_uploader("📂 Sube tu archivo de audio", type=["mp3", "wav", "m4a", "mp4", "aac"])

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp_path = None
    word_file = None

    try:
        # Guardar el archivo subido en un temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("🔄 Cargando modelo Whisper..."):
            model = load_model()

        with st.spinner("🔄 Transcribiendo audio, espera un momento..."):
            result = model.transcribe(tmp_path, language="es", fp16=False)

        transcription = result.get("text", "").strip()

        if not transcription:
            st.warning("⚠️ No se detectó texto en el audio. Verifica que el archivo tenga voz audible.")
        else:
            st.success("✅ Transcripción completa")
            st.subheader("📄 Texto transcrito:")
            st.write(transcription)

            # Generar Word en un archivo temporal para evitar conflictos entre sesiones
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_docx:
                word_file = tmp_docx.name

            doc = Document()
            doc.add_heading("Transcripción de Audio", 0)
            doc.add_paragraph(transcription)
            doc.save(word_file)

            with open(word_file, "rb") as f:
                st.download_button(
                    label="📥 Descargar Word",
                    data=f.read(),          # Leer a memoria antes de cerrar el archivo
                    file_name="transcripcion.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    except Exception as e:
        st.error(f"❌ Error al procesar el audio: {e}")

    finally:
        # Limpiar archivos temporales de forma segura
        for path in [tmp_path, word_file]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

