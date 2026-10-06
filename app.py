import streamlit as st
import chromadb
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Synora",
    page_icon="🤖"
)
st.markdown("""
<style>
.stApp {
    background: #07152f;
    color: white;
}
.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #38bdf8;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
}
.card {
    background: #10254a;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}
.stButton > button {
    width: 100%;
    background: #1683e8;
    color: white;
    border-radius: 10px;
    border: none;
}
</style>
""", unsafe_allow_html=True)
# ---------------- TITLE ----------------
st.markdown(
    '<div class="title">🤖 Synora</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Your AI PDF Assistant</div>',
    unsafe_allow_html=True
)
# ---------------- AI MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
model = load_model()
# ---------------- CHROMA ----------------
client = chromadb.PersistentClient(
    path="./chroma_db"
)
collection = client.get_or_create_collection(
    name="documents"
)
# ---------------- PDF ----------------
st.markdown("""
<div class="card">
<h2>📄 Upload PDF</h2>
<p>Upload a document and ask questions about it.</p>
</div>
""", unsafe_allow_html=True)
pdf = st.file_uploader(
    "Choose PDF",
    type="pdf"
)
# ---------------- PROCESS PDF ----------------
if pdf:
    if st.button("⚡ Process PDF"):
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        if not text.strip():
            st.error("Could not read text from this PDF.")
        else:
            # Split text into chunks
            chunks = [
                text[i:i + 1000]
                for i in range(0, len(text), 1000)
            ]
            # Create embeddings
            embeddings = model.encode(
                chunks
            ).tolist()
            # Store in ChromaDB
            ids = [
                f"chunk_{i}"
                for i in range(len(chunks))
            ]
            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings
            )
            st.success(
                f"✅ PDF processed! {len(chunks)} chunks created."
            )
# ---------------- QUESTION ----------------
st.markdown("""
<div class="card">
<h2>💬 Ask Synora</h2>
</div>
""", unsafe_allow_html=True)

question = st.text_input(
    "Ask something about your PDF",
    placeholder="Example: What is the main topic?"
)
# ---------------- ASK ----------------
if st.button("🚀 Ask Synora"):
    if not question:
        st.warning("Please enter a question.")
    elif collection.count() == 0:
        st.warning("Please upload and process a PDF first.")
    else:
        with st.spinner("🔎 Searching document..."):
            # Question embedding
            question_embedding = model.encode(
                [question]
            ).tolist()
            # Search ChromaDB
            results = collection.query(
                query_embeddings=question_embedding,
                n_results=3
            )
            documents = results["documents"][0]
        # Combine relevant text
        context = "\n\n".join(documents)
        # ---------------- Llama ----------------
        prompt = f"""
You are Synora, an AI assistant.
Answer the question using ONLY the information
from the document below.
If the answer is not available in the document,
say:
"I could not find the answer in the document."
DOCUMENT:
{context}
QUESTION:
{question}
ANSWER:
"""
        with st.spinner("🤖 Llama 3.2 is thinking..."):
            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        answer = response["message"]["content"]
        # ---------------- ANSWER ----------------
        st.markdown("""
        <div class="card">
        <h2>💡 Answer</h2>
        """, unsafe_allow_html=True)
        st.write(answer)
        st.markdown("</div>", unsafe_allow_html=True)
        