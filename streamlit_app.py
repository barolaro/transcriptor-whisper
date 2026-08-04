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

# Perfiles velocidad/calidad.
#   - "base" es el piso razonable: rápido y mucho mejor que "tiny" en español.
#   - "small" es el techo práctico en Streamlit Cloud gratuito (RAM ~1 GB):
#     mucho mejor calidad, un poco más lento.
# Se descartó "tiny" a propósito: es el que producía la mala calidad.
MODEL_PROFILES = {
    "Rápido": {
        "model": "base",
        "beam_size": 1,
        "best_of": 1,
        "condition_on_previous_text": False,
        "descripcion": "Más veloz. Bien para audio limpio, una sola voz.",
    },
    "Recomendado · mejor calidad": {
        "model": "small",
        "beam_size": 5,
        "best_of": 5,
        "condition_on_previous_text": True,
        "descripcion": "Bastante mejor con ruido, varias voces o términos técnicos.",
    },
}

# Pista inicial: orienta al modelo hacia español con puntuación y mayúsculas
# correctas. Es "gratis" (no cuesta velocidad) y mejora acentos y formato.
INITIAL_PROMPT = (
    "Transcripción en español de Chile, con puntuación, acentos y "
    "mayúsculas correctas."
)


@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> WhisperModel:
    """Carga una sola vez el modelo optimizado para CPU (INT8)."""
    cpu_threads = max(1, min(4, os.cpu_count() or 1))
    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
        num_workers=1,
    )


def transcribe_stream(audio_bytes: bytes, suffix: str, profile: dict):
    """Transcribe emitiendo cada segmento a medida que se completa.

    Es un generador: entrega (texto_acumulado, progreso 0-1). Esto permite
    mostrar avance real y evita la sensación de que la app quedó congelada.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        model = load_model(profile["model"])
        segments, info = model.transcribe(
            temp_path,
            language="es",
            task="transcribe",
            beam_size=profile["beam_size"],
            best_of=profile["best_of"],
            # Fallback de temperatura: si un tramo sale con baja confianza o
            # texto "basura", reintenta con más aleatoriedad. Sube robustez.
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=profile["condition_on_previous_text"],
            initial_prompt=INITIAL_PROMPT,
            vad_filter=True,
            # speech_pad_ms=400 evita cortar la primera/última palabra de cada
            # tramo de voz: causa frecuente de palabras "comidas".
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400,
            },
        )

        total = float(info.duration) or 0.0
        parts: list[str] = []
        for seg in segments:
            parts.append(seg.text.strip())
            progress = min(1.0, seg.end / total) if total else 0.0
            yield " ".join(parts).strip(), progress, total

        # Garantiza un último yield con progreso completo aunque no haya voz.
        yield " ".join(parts).strip(), 1.0, total
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
        st.session_state.pop("result_key", None)


st.title("🎙️ Transcriptor de Audio")
st.write(
    "Sube un audio y obtén su transcripción en español. "
    "El procesamiento comienza solo cuando presionas **Transcribir**."
)

model_label = st.radio(
    "Velocidad / calidad",
    options=list(MODEL_PROFILES),
    index=1,  # por defecto: mejor calidad
    horizontal=True,
)
st.caption(MODEL_PROFILES[model_label]["descripcion"])

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

    profile = MODEL_PROFILES[model_label]
    # La clave incluye el modelo: si cambias de perfil, no reusa un resultado
    # de otra calidad, pero sí evita recalcular al descargar.
    result_key = f"{file_id}:{profile['model']}:{profile['beam_size']}"

    if st.button("🚀 Transcribir audio", type="primary", use_container_width=True):
        suffix = Path(uploaded_file.name).suffix.lower() or ".audio"
        try:
            st.info(
                "La primera vez puede tardar más mientras se descarga y carga "
                "el modelo. Las siguientes son más rápidas."
            )
            bar = st.progress(0.0, text="Preparando el modelo…")
            preview = st.empty()

            final_text = ""
            duration = 0.0
            for text, progress, total in transcribe_stream(
                audio_bytes, suffix, profile
            ):
                final_text = text
                duration = total
                bar.progress(
                    progress,
                    text=f"Transcribiendo… {progress * 100:.0f}%",
                )
                if text:
                    preview.text_area(
                        "Avance",
                        value=text,
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                    )

            bar.progress(1.0, text="Transcripción finalizada ✅")
            preview.empty()

            st.session_state.transcription = final_text
            st.session_state.duration = duration
            st.session_state.result_key = result_key
        except Exception as exc:
            st.error(
                "No fue posible procesar el audio. Comprueba que el archivo "
                "no esté dañado e inténtalo nuevamente."
            )
            st.exception(exc)

    # Se muestra el resultado solo si corresponde al perfil/archivo actual.
    transcription = st.session_state.get("transcription")
    if transcription and st.session_state.get("result_key") == result_key:
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
    elif (
        st.session_state.get("result_key") == result_key
        and "transcription" in st.session_state
    ):
        st.warning(
            "No se detectó voz. Prueba con el modo Recomendado o revisa el "
            "volumen del audio."
        )
