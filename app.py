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

def dividir_texto(texto, max_tokens=500):
    """Divide el texto en fragmentos más pequeños de manera inteligente."""
    tokens = texto.split()
    fragmentos = []
    fragmento_actual = []
    cuenta_tokens = 0
    
    for token in tokens:
      if cuenta_tokens + len(token.split()) <= max_tokens:
        fragmento_actual.append(token)
        cuenta_tokens += len(token.split())
      else:
         fragmentos.append(" ".join(fragmento_actual))
         fragmento_actual = [token]
         cuenta_tokens = len(token.split())

    if fragmento_actual:
        fragmentos.append(" ".join(fragmento_actual))
    return fragmentos

def limpiar_transcripcion_gemini(texto, max_retries=3, initial_delay=1):
    """Limpia una transcripción usando Gemini con reintentos."""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
     **Quiero que te transformes en un orador experto**, capaz de cautivar a una audiencia numerosa con una charla profesional y reflexiva.  

Tu discurso debe abordar temas profundos y universales, despertando el interés del público e invitando a la reflexión. Debe transmitir ideas con un lenguaje persuasivo, evocador y natural, como si estuvieras compartiendo una experiencia propia o contando una historia real.  

### **Instrucciones:**  
- **Reescribe el texto** con tus propias palabras, asegurándote de que suene fluido, natural y humano.  
- **Expande las ideas cuando sea necesario**, enriqueciendo la narración con ejemplos, matices y reflexiones adicionales.  
- **Evita frases rígidas o mecánicas.** Debe parecer que el texto ha sido escrito por un experto en comunicación y no por una IA.  
- **No incluyas nombres propios, libros ni fuentes específicas.** En su lugar, refiérete a ellos de manera general.  
- **Mantén una extensión similar o mayor** al texto original. No omitas información ni reduzcas su profundidad.  
- **Adapta la estructura para que la narración fluya** como un discurso atractivo, sin parecer un resumen o una simple reformulación.  
- **Elimina cualquier asterisco, formato adicional o estructura que delate que el texto fue generado artificialmente.**  
- **Optimiza el contenido para que suene bien en voz alta,** haciendo que cada frase fluya de forma natural y sin pausas forzadas.  

**Objetivo:** Que el texto resultante parezca una obra original de un gran orador, logrando que nadie sospeche que ha sido generado artificialmente.
        {texto}

        Texto corregido:
    """
    retries = 0
    delay = initial_delay
    while retries <= max_retries:
        try:
            logging.info(f"Enviando solicitud a Gemini para texto: {texto[:50]}... (Intento {retries + 1})")
            response = model.generate_content(prompt)
            if response.text:
              logging.info(f"Respuesta recibida de Gemini para texto: {texto[:50]}")
              return response.text
            else:
              logging.error(f"Respuesta vacía de Gemini para el texto: {texto[:50]}. (Intento {retries + 1})")
              retries += 1
              sleep(delay)
              delay *= 2
        except Exception as e:
            logging.error(f"Error en la solicitud a Gemini: {e} (Intento {retries + 1})")
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
