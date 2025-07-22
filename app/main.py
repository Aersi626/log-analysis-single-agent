import os
import time
from datetime import datetime, timezone
import requests
import streamlit as st
import logging

from elasticsearch import Elasticsearch
from langchain_community.vectorstores import ElasticsearchStore
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration from Environment Variables ---
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
ELASTICSEARCH_USER = os.getenv("ELASTICSEARCH_USER", None)
ELASTICSEARCH_PASS = os.getenv("ELASTICSEARCH_PASS", None)
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://host.docker.internal:11434")
EMBEDDING_API_BASE_URL = os.getenv("EMBEDDING_API_BASE_URL", "http://embedding_service:8080")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LOG_INDEX_NAME = os.getenv("LOG_INDEX_NAME", "your_log_index")

MAX_CONNECTION_ATTEMPTS = 120
RETRY_DELAY_SECONDS = 5

USE_SSL = ELASTICSEARCH_URL.startswith("https")

# --- 1. Elasticsearch Client Initialization ---
@st.cache_resource
def get_es_client():
    for i in range(1, MAX_CONNECTION_ATTEMPTS + 1):
        try:
            st.info(f"Attempt {i}/{MAX_CONNECTION_ATTEMPTS}: Connecting to Elasticsearch at {ELASTICSEARCH_URL}...")
            logger.info(f"Attempt {i}: Connecting to {ELASTICSEARCH_URL}")

            es = Elasticsearch(
                ELASTICSEARCH_URL,
                verify_certs=False,      # explicitly turn off SSL verification
                ssl_show_warn=False,     # suppress SSL warnings
                request_timeout=60
            )

            # Use info() instead of ping() for a real check
            info = es.info()
            logger.info(f"Elasticsearch info: {info.body}")
            st.success("✅ Connected to Elasticsearch!")
            return es

        except Exception as e:
            st.warning(f"⚠️ Attempt {i} failed: {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
            logger.error(f"Connection attempt {i} failed: {e}", exc_info=True)
            time.sleep(RETRY_DELAY_SECONDS)

    st.error(f"❌ Failed to connect to Elasticsearch after {MAX_CONNECTION_ATTEMPTS} attempts.")
    logger.critical("Could not connect to Elasticsearch.")
    st.stop()

es = get_es_client()

# --- 2. Embedding Model ---
class RemoteHuggingFaceEmbeddings:
    def __init__(self, api_base_url: str, model_name: str):
        self.api_base_url = api_base_url
        self.model_name = model_name
        self._ensure_service_ready()

    def _ensure_service_ready(self):
        for i in range(1, MAX_CONNECTION_ATTEMPTS + 1):
            try:
                st.info(f"Attempt {i}: Checking Embedding Service at {self.api_base_url}...")
                response = requests.get(f"{self.api_base_url}/health", timeout=10)
                response.raise_for_status()
                st.success(f"✅ Embedding Service ready at {self.api_base_url} with model {self.model_name}")
                logger.info("Embedding service is ready.")
                return
            except requests.exceptions.RequestException as e:
                st.warning(f"⚠️ Embedding Service not ready: {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                logger.warning(f"Embedding service check failed: {e}")
                time.sleep(RETRY_DELAY_SECONDS)

        st.error("❌ Could not connect to Embedding Service.")
        logger.critical("Could not connect to Embedding Service.")
        st.stop()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"inputs": texts}
        response = requests.post(
            f"{self.api_base_url}/embed",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        embeddings = response.json()
        if not isinstance(embeddings, list):
            raise ValueError(f"Unexpected response from embedding service: {embeddings}")
        if not embeddings:
            raise ValueError("Embedding service returned empty embeddings.")
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

embedding_model = RemoteHuggingFaceEmbeddings(
    api_base_url=EMBEDDING_API_BASE_URL,
    model_name=EMBEDDING_MODEL
)

# --- 3. Ensure Elasticsearch Index ---
@st.cache_resource
def create_index_if_not_exists(_es_client_instance, index_name):
    if not _es_client_instance.indices.exists(index=index_name):
        st.info(f"Index '{index_name}' does not exist. Creating...")
        mapping = {
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "log_level": {"type": "keyword"},
                    "service_name": {"type": "keyword"},
                    "message": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384
                    }
                }
            }
        }
        _es_client_instance.indices.create(index=index_name, body=mapping)
        st.success(f"✅ Index '{index_name}' created.")
        logger.info(f"Index '{index_name}' created.")
    else:
        logger.info(f"Index '{index_name}' already exists.")

create_index_if_not_exists(es, LOG_INDEX_NAME)

# --- 4. LangChain Setup ---
vectorstore = ElasticsearchStore(
    es_connection=es,
    index_name=LOG_INDEX_NAME,
    embedding=embedding_model,
    vector_query_field="embedding",)

logger.info("Vectorstore & retriever initialized.")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_API_BASE_URL)
logger.info(f"Ollama model '{OLLAMA_MODEL}' initialized.")

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an expert system administrator and software engineer. Analyze the provided log snippets, identify errors, root cause, and actionable fixes. If no errrors are found, summarize the logs and suggest improvements. 
    If insufficient context, say so. Use concise, bullet-pointed output.
    Context:
    {context}
    """),
    ("user", "User Query: {input}\n\nAnalysis, Root Cause, and Fix:")
])

Youtube_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, Youtube_chain)

# --- Streamlit UI ---
st.set_page_config(page_title="LLM Log Analyzer", layout="wide")
st.title("💡 LLM-Powered Log Analysis")
st.markdown("""
This tool uses an LLM + Elasticsearch to analyze logs and suggest fixes.
""")

user_query = st.text_area("Your Log Query:", height=100,
                          placeholder="e.g., Why is service X failing?")

if st.button("Analyze Logs", type="primary"):
    if user_query:
        with st.spinner("Analyzing..."):
            try:
                logger.info(f"User query: '{user_query}'")
                response = rag_chain.invoke({"input": user_query})

                st.subheader("🔍 Analysis Results:")
                st.markdown(response["answer"])

                st.subheader("📄 Relevant Logs:")
                if response["context"]:
                    for i, doc in enumerate(response["context"]):
                        st.text(f"--- Log Snippet {i+1} ---")
                        display_data = {"message": doc.page_content, **doc.metadata}
                        st.json(display_data)
                else:
                    st.info("No relevant logs found.")
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                logger.error(f"Error in RAG chain: {e}", exc_info=True)
                st.exception(e)
    else:
        st.warning("Please enter a query.")

# --- Manual Log Ingestion ---
st.sidebar.title("📥 Manual Log Ingestion")
with st.sidebar.form("log_form"):
    log_message = st.text_area("Log Message:", height=150)
    log_level = st.selectbox("Log Level:", ["INFO", "WARN", "ERROR", "DEBUG"], index=2)
    service_name = st.text_input("Service Name:", "test-service")
    current_timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds') + 'Z'
    st.write(f"Timestamp: `{current_timestamp}`")

    if st.form_submit_button("Add Log"):
        if log_message:
            try:
                embedding = embedding_model.embed_query(log_message)
                log_doc = {
                    "timestamp": current_timestamp,
                    "log_level": log_level,
                    "message": log_message,
                    "service_name": service_name,
                    "embedding": embedding
                }
                resp = es.index(index=LOG_INDEX_NAME, document=log_doc)
                st.sidebar.success(f"Log indexed! ID: {resp['_id']}")
                st.sidebar.json(log_doc)
            except Exception as e:
                st.sidebar.error(f"Failed to ingest log: {e}")
                logger.error(f"Ingestion error: {e}", exc_info=True)
        else:
            st.sidebar.warning("Enter a log message.")

st.sidebar.markdown("---")
st.sidebar.info("""
**Setup Notes:**
✅ Elasticsearch index exists with `dense_vector`.  
✅ Ollama running on host.  
✅ Embedding service running & healthy.  
Restart app after config changes!
""")
