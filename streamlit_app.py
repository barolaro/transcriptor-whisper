import hashlib
import io
import os
import tempfile
from pathlib import Path

import streamlit as st
from docx import Document
from faster_whisper import WhisperModel


st.set_page_config(
    page_title="Transcriptor de Audio",
    page_icon="🎙️",
    layout="centered",
)

SUPPORTED_FORMATS = ["mp3", "wav", "m4a", "mp4", "aac", "ogg", "webm"]
MODEL_OPTIONS = {
    "Rápido (recomendado)": "tiny",
    "Mayor precisión": "base",
}


@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> WhisperModel:
    """Carga una sola vez el modelo optimizado para CPU."""
    cpu_threads = max(1, min(4, os.cpu_count() or 1))
    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
        num_workers=1,
    )


@st.cache_data(show_spinner=False, max_entries=20, ttl=86_400)
def transcribe_audio(
    audio_bytes: bytes,
    suffix: str,
    model_name: str,
) -> tuple[str, float]:
    """Transcribe y guarda temporalmente resultados idénticos por 24 horas."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        model = load_model(model_name)
        segments, info = model.transcribe(
            temp_path,
            language="es",
            beam_size=1,
            best_of=1,
            temperature=0,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            without_timestamps=True,
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, float(info.duration)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def create_word_document(text: str, original_name: str) -> bytes:
    document = Document()
    document.add_heading("Transcripción de audio", 0)
    document.add_paragraph(f"Archivo: {original_name}")
    document.add_paragraph(text)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def reset_result_if_file_changed(file_id: str) -> None:
    if st.session_state.get("current_file_id") != file_id:
        st.session_state.current_file_id = file_id
        st.session_state.pop("transcription", None)
        st.session_state.pop("duration", None)


st.title("🎙️ Transcriptor de Audio")
st.write(
    "Sube un audio y obtén su transcripción en español. "
    "El procesamiento comienza solo cuando presionas **Transcribir**."
)

model_label = st.radio(
    "Velocidad de transcripción",
    options=list(MODEL_OPTIONS),
    horizontal=True,
    help=(
        "El modo rápido consume menos CPU. Usa Mayor precisión si el audio "
        "tiene ruido, varias voces o términos técnicos."
    ),
)

uploaded_file = st.file_uploader(
    "📂 Sube tu archivo de audio",
    type=SUPPORTED_FORMATS,
)

if uploaded_file is not None:
    audio_bytes = uploaded_file.getvalue()
    file_id = hashlib.sha256(audio_bytes).hexdigest()
    reset_result_if_file_changed(file_id)

    size_mb = len(audio_bytes) / (1024 * 1024)
    st.caption(f"{uploaded_file.name} · {size_mb:.1f} MB")

    if size_mb > 200:
        st.error("El archivo supera el límite de 200 MB.")
        st.stop()

    if st.button("🚀 Transcribir audio", type="primary", use_container_width=True):
        suffix = Path(uploaded_file.name).suffix.lower() or ".audio"
        selected_model = MODEL_OPTIONS[model_label]

        try:
            with st.status(
                "Preparando el modelo y transcribiendo…",
                expanded=True,
            ) as status:
                st.write(
                    "La primera transcripción puede tardar más mientras se "
                    "descarga y carga el modelo."
                )
                transcription, duration = transcribe_audio(
                    audio_bytes,
                    suffix,
                    selected_model,
                )
                status.update(
                    label="Transcripción finalizada",
                    state="complete",
                    expanded=False,
                )

            st.session_state.transcription = transcription
            st.session_state.duration = duration
        except Exception as exc:
            st.error(
                "No fue posible procesar el audio. Comprueba que el archivo "
                "no esté dañado e inténtalo nuevamente."
            )
            st.exception(exc)

    transcription = st.session_state.get("transcription")
    if transcription:
        duration = st.session_state.get("duration", 0)
        st.success(f"✅ Listo. Audio procesado: {duration / 60:.1f} minutos.")
        st.subheader("📄 Texto transcrito")
        edited_text = st.text_area(
            "Puedes corregir el texto antes de descargarlo:",
            value=transcription,
            height=320,
            label_visibility="collapsed",
        )

        base_name = Path(uploaded_file.name).stem
        word_bytes = create_word_document(edited_text, uploaded_file.name)

        col1, col2 = st.columns(2)
        col1.download_button(
            "📥 Descargar TXT",
            data=edited_text.encode("utf-8"),
            file_name=f"{base_name}_transcripcion.txt",
            mime="text/plain",
            use_container_width=True,
        )
        col2.download_button(
            "📥 Descargar Word",
            data=word_bytes,
            file_name=f"{base_name}_transcripcion.docx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            use_container_width=True,
        )
    elif "transcription" in st.session_state:
        st.warning(
            "No se detectó voz. Prueba con el modo Mayor precisión o revisa "
            "el volumen del audio."
        )
