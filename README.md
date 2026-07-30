# Transcriptor de Audio

Aplicación Streamlit para transcribir audios en español y descargar el
resultado en TXT o Word.

## Mejoras de rendimiento

- Usa `faster-whisper` con cuantización INT8, optimizada para CPU.
- Procesa el audio directamente, sin crear una copia WAV de gran tamaño.
- El modo **Rápido** usa el modelo `tiny`; el modo **Mayor precisión** usa
  `base`.
- El modelo queda cargado en caché y las transcripciones idénticas se
  reutilizan durante 24 horas.
- La transcripción comienza con un botón, evitando que Streamlit repita el
  proceso al descargar o interactuar con la página.
- El filtro VAD omite tramos sin voz.

## Uso

1. Sube un archivo MP3, WAV, M4A, MP4, AAC, OGG o WEBM.
2. Elige el modo de velocidad.
3. Presiona **Transcribir audio**.
4. Corrige el texto si es necesario y descárgalo en TXT o Word.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

> Streamlit Community Cloud puede limitar temporalmente la CPU de una
> aplicación. Esta versión reduce mucho el consumo, pero no puede eliminar una
> limitación aplicada por la propia plataforma.

Desarrollado por Bayron Retamal.
