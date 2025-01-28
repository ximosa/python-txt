import streamlit as st
import os
import concurrent.futures
from time import perf_counter, sleep
import logging
import google.generativeai as genai

st.set_page_config(
    page_title="Texto Corto",
    layout="wide"
)

# Configuración de Logging
LOG_FILE = 'app.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Obtener la API Key de Gemini
try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
except KeyError:
    st.error("La variable de entorno GEMINI_API_KEY no está configurada.")
    st.stop()

def dividir_texto(texto, max_tokens=2000, overlap_tokens=200):
    """Divide el texto en fragmentos con solapamiento."""
    tokens = texto.split()
    fragmentos = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        fragmento = " ".join(tokens[start:end])
        fragmentos.append(fragmento)
        start = end - overlap_tokens  # Ajustar el inicio para solapamiento
    return fragmentos

def limpiar_transcripcion_gemini(texto, max_retries=3, initial_delay=1):
    """Limpia una transcripción usando Gemini con reintentos."""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    
    prompt_planificacion = f"""
        Actúa como un narrador personal y reflexivo, compartiendo tus pensamientos sobre el texto que te voy a dar. Escribe como si fueras el autor del texto, pero con tus propias palabras.

        Sigue estas pautas para planificar tu texto:
        - Analiza el texto de entrada.
        - Determina los puntos principales que debes cubrir.
        - Asegúrate de que la longitud del texto resultante sea **al menos igual** a la del texto original.
        - Planifica la estructura general del texto, incluyendo el título.
        - Describe el tono del texto y la forma en que vas a narrar los hechos.
        - Haz un borrador del texto que vas a escribir, para tener una idea de como quedara el texto final.
        - **Importante: No reduzcas la cantidad de información ni la longitud del texto. El texto generado debe ser de longitud similar o superior al texto de entrada.**

        {texto}
        
        Planificación del texto:
        """
    
    retries = 0
    delay = initial_delay
    while retries <= max_retries:
        try:
            logging.info(f"Enviando solicitud de planificación a Gemini para texto: {texto[:50]}... (Intento {retries + 1})")
            response_planificacion = model.generate_content(prompt_planificacion)
            if response_planificacion.text:
              planificacion = response_planificacion.text
              logging.info(f"Planificación recibida de Gemini para texto: {texto[:50]}")
            else:
              logging.error(f"Respuesta vacía de planificación de Gemini para el texto: {texto[:50]}. (Intento {retries + 1})")
              retries += 1
              sleep(delay)
              delay *= 2
              continue
        except Exception as e:
            logging.error(f"Error en la solicitud de planificación a Gemini: {e} (Intento {retries + 1})")
            retries += 1
            sleep(delay)
            delay *= 2
            continue
            
        prompt_generacion = f"""
            Actúa como un narrador personal y reflexivo, compartiendo tus pensamientos sobre el texto que te voy a dar. Escribe como si fueras el autor del texto, pero con tus propias palabras.

            Sigue estas pautas:
            - Usa el plan que creaste anteriormente, y escribe el texto completo con tus propias palabras, expandiendo cada idea si es necesario.
            - Asegúrate de que la longitud del texto resultante sea **al menos igual** a la del texto original.
            - Proporciona un título atractivo que capture la esencia del texto.
            - Evita menciones directas de personajes o del autor.
            - Concéntrate en transmitir la experiencia general, las ideas principales, los temas y las emociones.
            - Usa un lenguaje personal y evocador, como si estuvieras compartiendo tus propias conclusiones después de una reflexión profunda.
            - Evita nombres propios o lugares específicos.
            - Narra los hechos como si fueran una historia.
            - Elimina cualquier asterisco o formato adicional, incluyendo negritas o encabezados.
            - Asegúrate de que el texto sea apto para la lectura con voz de Google.
            - **Importante: No reduzcas la cantidad de información ni la longitud del texto. El texto generado debe ser de longitud similar o superior al texto de entrada.**

            Planificación: {planificacion}

            Texto corregido:
            """
            
        try:
            logging.info(f"Enviando solicitud de generación a Gemini para texto: {texto[:50]}... (Intento {retries + 1})")
            response_generacion = model.generate_content(prompt_generacion)
            if response_generacion.text:
                logging.info(f"Respuesta de generación recibida de Gemini para texto: {texto[:50]}")
                return response_generacion.text
            else:
                logging.error(f"Respuesta vacía de generación de Gemini para el texto: {texto[:50]}. (Intento {retries + 1})")
                retries += 1
                sleep(delay)
                delay *= 2
        except Exception as e:
            logging.error(f"Error en la solicitud de generación a Gemini: {e} (Intento {retries + 1})")
            retries += 1
            sleep(delay)
            delay *= 2
    
    logging.error(f"Máximo número de reintentos alcanzado para el texto: {texto[:50]}.")
    return None


def procesar_transcripcion(texto):
    """Procesa el texto dividiendo en fragmentos y usando Gemini en paralelo."""
    fragmentos = dividir_texto(texto)
    texto_limpio_completo = ""
    total_fragmentos = len(fragmentos)
    progress_bar = st.progress(0)

    with st.spinner("Procesando con Gemini..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(limpiar_transcripcion_gemini, fragmento): fragmento
                for fragmento in fragmentos
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                texto_limpio = future.result()
                if texto_limpio:
                    texto_limpio_completo += texto_limpio + " "
                progress_bar.progress(i / total_fragmentos)

    return texto_limpio_completo.strip()


def descargar_texto(texto_formateado):
    """Genera un enlace de descarga para el texto formateado."""
    return st.download_button(
        label="Descargar Texto",
        data=texto_formateado.encode('utf-8'),
        file_name="transcripcion_formateada.txt",
        mime="text/plain"
    )

def mostrar_logs():
    """Muestra los logs en Streamlit."""
    try:
      with open(LOG_FILE, 'r', encoding='utf-8') as f:
          log_content = f.read()
          st.subheader("Logs de la Aplicación:")
          st.text_area("Logs", value=log_content, height=300)
    except FileNotFoundError:
      st.error("El archivo de logs no fue encontrado.")

st.title("Limpiador de Transcripciones de YouTube (con Gemini)")

transcripcion = st.text_area("Pega aquí tu transcripción sin formato:")

if 'texto_procesado' not in st.session_state:
    st.session_state['texto_procesado'] = ""

if st.button("Procesar"):
    if transcripcion:
        start_time = perf_counter()
        texto_limpio = procesar_transcripcion(transcripcion)
        end_time = perf_counter()
        st.session_state['texto_procesado'] = texto_limpio
        st.success(f"Tiempo de procesamiento: {end_time - start_time:.2f} segundos")
    else:
        st.warning("Por favor, introduce el texto a procesar.")

if st.session_state['texto_procesado']:
    st.subheader("Transcripción Formateada:")
    st.write(st.session_state['texto_procesado'])
    descargar_texto(st.session_state['texto_procesado'])

if st.checkbox("Mostrar Logs"):
  mostrar_logs()
