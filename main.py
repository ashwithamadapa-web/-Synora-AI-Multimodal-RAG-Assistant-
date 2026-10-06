import streamlit as st
import chromadb
import ollama
import fitz
import hashlib
import os
import base64

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Synora AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "image_paths" not in st.session_state:
    st.session_state.image_paths = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Synora AI")

st.subheader("Multimodal RAG Assistant")

st.write(
    "📚 Ask questions about your PDFs using "
    "Retrieval-Augmented Generation, semantic search "
    "and AI-powered visual understanding."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Synora Settings")

    st.divider()

    st.subheader("🤖 AI Model")

    text_model = st.selectbox(
        "Text Model",
        [
            "llama3.2",
            "llama3.1",
            "llama3"
        ],
        index=0
    )

    vision_model = st.selectbox(
        "Vision Model",
        [
            "llama3.2-vision"
        ]
    )

    st.divider()

    st.subheader("🔎 Retrieval")

    top_k = st.slider(
        "Number of chunks",
        min_value=1,
        max_value=10,
        value=5
    )

    st.divider()

    st.subheader("🗄️ Knowledge Base")

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="synora_documents"
    )

    chunk_count = collection.count()

    st.metric(
        "Text Chunks",
        chunk_count
    )

    st.metric(
        "Documents",
        len(st.session_state.processed_files)
    )

    st.metric(
        "Images",
        len(st.session_state.image_paths)
    )

    st.divider()

    if st.button(
        "🧹 Clear Knowledge Base",
        use_container_width=True
    ):

        try:

            client.delete_collection(
                "synora_documents"
            )

            collection = client.get_or_create_collection(
                name="synora_documents"
            )

            st.session_state.processed_files = {}
            st.session_state.image_paths = []
            st.session_state.chat_history = []

            st.success(
                "Knowledge base cleared."
            )

            st.rerun()

        except Exception as e:

            st.error(str(e))


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


with st.spinner("Loading AI embedding model..."):

    embedding_model = load_embedding_model()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def file_hash(file_bytes):

    return hashlib.md5(
        file_bytes
    ).hexdigest()


# ------------------------------------------------------------
# PDF TEXT EXTRACTION
# ------------------------------------------------------------

def extract_pdf_text(pdf_bytes):

    reader = PdfReader(
        __import__("io").BytesIO(pdf_bytes)
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages
    ):

        text = page.extract_text()

        if text and text.strip():

            pages.append(
                {
                    "page": page_number + 1,
                    "text": text
                }
            )

    return pages


# ------------------------------------------------------------
# TEXT CHUNKING
# ------------------------------------------------------------

def split_text(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

        start += chunk_size - overlap

    return chunks


# ------------------------------------------------------------
# IMAGE EXTRACTION
# ------------------------------------------------------------

def extract_images(
    pdf_bytes,
    document_hash,
    document_name
):

    image_folder = "synora_images"

    os.makedirs(
        image_folder,
        exist_ok=True
    )

    images = []

    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    for page_index in range(
        len(pdf)
    ):

        page = pdf[page_index]

        page_images = page.get_images(
            full=True
        )

        for image_index, image in enumerate(
            page_images
        ):

            xref = image[0]

            extracted = pdf.extract_image(
                xref
            )

            image_bytes = extracted[
                "image"
            ]

            extension = extracted[
                "ext"
            ]

            filename = (
                f"{document_hash}_"
                f"page_{page_index + 1}_"
                f"image_{image_index + 1}."
                f"{extension}"
            )

            path = os.path.join(
                image_folder,
                filename
            )

            with open(
                path,
                "wb"
            ) as f:

                f.write(image_bytes)

            images.append(
                {
                    "path": path,
                    "page": page_index + 1,
                    "file": document_name
                }
            )

    pdf.close()

    return images


# ============================================================
# UPLOAD SECTION
# ============================================================

st.header("📄 Document Center")

st.write(
    "Upload one or more PDF documents to build "
    "your private knowledge base."
)

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS BUTTON
# ============================================================

if uploaded_files:

    st.info(
        f"📚 {len(uploaded_files)} document(s) selected."
    )

    if st.button(
        "⚡ Process Documents",
        use_container_width=True,
        type="primary"
    ):

        progress = st.progress(
            0
        )

        status = st.empty()

        total_files = len(
            uploaded_files
        )

        for file_index, uploaded_file in enumerate(
            uploaded_files
        ):

            file_bytes = uploaded_file.getvalue()

            document_hash = file_hash(
                file_bytes
            )

            document_name = (
                uploaded_file.name
            )

            if document_hash in st.session_state.processed_files:

                status.info(
                    f"Skipping {document_name} — already processed."
                )

                progress.progress(
                    (file_index + 1) / total_files
                )

                continue


            status.info(
                f"📖 Reading {document_name}..."
            )


            # ------------------------------------------------
            # EXTRACT TEXT
            # ------------------------------------------------

            pages = extract_pdf_text(
                file_bytes
            )

            chunks = []

            for page_data in pages:

                page_chunks = split_text(
                    page_data["text"]
                )

                for chunk in page_chunks:

                    chunks.append(
                        {
                            "text": chunk,
                            "page": page_data["page"]
                        }
                    )


            # ------------------------------------------------
            # EMBEDDINGS
            # ------------------------------------------------

            if chunks:

                status.info(
                    f"🧠 Creating embeddings for "
                    f"{document_name}..."
                )

                texts = [
                    x["text"]
                    for x in chunks
                ]

                embeddings = (
                    embedding_model
                    .encode(texts)
                    .tolist()
                )


                ids = []

                metadata = []


                for i, chunk in enumerate(
                    chunks
                ):

                    ids.append(
                        f"{document_hash}_chunk_{i}"
                    )

                    metadata.append(
                        {
                            "file": document_name,
                            "page": chunk["page"],
                            "type": "text"
                        }
                    )


                collection.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadata
                )


            # ------------------------------------------------
            # EXTRACT IMAGES
            # ------------------------------------------------

            status.info(
                f"🖼️ Extracting images from "
                f"{document_name}..."
            )

            images = extract_images(
                file_bytes,
                document_hash,
                document_name
            )

            st.session_state.image_paths.extend(
                images
            )


            # ------------------------------------------------
            # SAVE DOCUMENT INFO
            # ------------------------------------------------

            st.session_state.processed_files[
                document_hash
            ] = {
                "name": document_name,
                "pages": len(pages),
                "chunks": len(chunks),
                "images": len(images)
            }


            progress.progress(
                (file_index + 1) / total_files
            )


        status.success(
            "✅ All documents processed successfully!"
        )

        st.balloons()


# ============================================================
# DOCUMENT LIBRARY
# ============================================================

if st.session_state.processed_files:

    st.divider()

    st.header("📚 Document Library")

    for document in (
        st.session_state
        .processed_files
        .values()
    ):

        with st.expander(
            f"📄 {document['name']}"
        ):

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Pages",
                    document["pages"]
                )

            with col2:

                st.metric(
                    "Chunks",
                    document["chunks"]
                )

            with col3:

                st.metric(
                    "Images",
                    document["images"]
                )


# ============================================================
# ASK SYNORA
# ============================================================

st.divider()

st.header("💬 Ask Synora")

st.write(
    "Ask questions about your uploaded documents."
)


question = st.text_input(
    "Your question",
    placeholder=(
        "Example: What are the main conclusions "
        "of this document?"
    )
)


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.write("💡 Suggested questions")

q1, q2, q3 = st.columns(3)

with q1:

    if st.button(
        "📌 Summarize the document"
    ):

        question = (
            "Give me a concise summary "
            "of the document."
        )

with q2:

    if st.button(
        "🔑 Main points"
    ):

        question = (
            "What are the most important "
            "points in the document?"
        )

with q3:

    if st.button(
        "📊 Explain the figures"
    ):

        question = (
            "Explain the figures, charts "
            "and diagrams in the document."
        )


# ============================================================
# TEXT ANSWER
# ============================================================

def generate_answer(
    question,
    context
):

    prompt = f"""
You are Synora AI, a professional
multimodal PDF assistant.

Answer the user's question using
ONLY the supplied document context.

Rules:

1. Do not invent information.
2. Use information from the context.
3. If the answer is unavailable,
   clearly say that it was not found.
4. Give a clear and structured answer.
5. Mention important page numbers when available.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""


    response = ollama.chat(
        model=text_model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response[
        "message"
    ][
        "content"
    ]


# ============================================================
# VISION ANALYSIS
# ============================================================

def analyze_image(
    image_path,
    question
):

    prompt = f"""
You are Synora AI Vision.

Analyze the supplied PDF image.

Answer the user's question based
only on what is visible in the image.

Question:

{question}

Describe relevant charts,
diagrams, tables, labels,
numbers and relationships
when appropriate.
"""


    response = ollama.chat(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [
                    image_path
                ]
            }
        ]
    )

    return response[
        "message"
    ][
        "content"
    ]


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🔍 Ask Synora",
    use_container_width=True,
    type="primary"
):

    if not question:

        st.warning(
            "Please enter a question."
        )

    elif collection.count() == 0:

        st.warning(
            "Please upload and process "
            "a PDF first."
        )

    else:

        # ----------------------------------------------------
        # EMBEDDING SEARCH
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Searching your documents..."
        ):

            query_embedding = (
                embedding_model
                .encode([question])
                .tolist()
            )


            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )


        documents = results[
            "documents"
        ][0]

        metadatas = results[
            "metadatas"
        ][0]


        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context_parts = []

        sources = []


        for document, metadata in zip(
            documents,
            metadatas
        ):

            context_parts.append(
                f"""
File: {metadata['file']}

Page: {metadata['page']}

Content:
{document}
"""
            )

            sources.append(
                {
                    "file": metadata[
                        "file"
                    ],
                    "page": metadata[
                        "page"
                    ]
                }
            )


        context = "\n\n".join(
            context_parts
        )


        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Synora is thinking..."
        ):

            answer = generate_answer(
                question,
                context
            )


        st.session_state.last_answer = (
            answer
        )

        st.session_state.last_sources = (
            sources
        )


        # ----------------------------------------------------
        # SAVE CHAT
        # ----------------------------------------------------

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )


        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        st.divider()

        st.subheader(
            "🤖 Synora's Answer"
        )

        st.write(answer)


        # ====================================================
        # SOURCES
        # ====================================================

        st.subheader(
            "📌 Sources"
        )

        unique_sources = []

        for source in sources:

            source_text = (
                f"📄 {source['file']} "
                f"— Page {source['page']}"
            )

            if source_text not in unique_sources:

                unique_sources.append(
                    source_text
                )

        for source in unique_sources:

            st.caption(source)


        # ====================================================
        # DOWNLOAD ANSWER
        # ====================================================

        st.download_button(
            "📥 Download Answer",
            data=answer,
            file_name="synora_answer.txt",
            mime="text/plain"
        )


        # ====================================================
        # IMAGE / VISION ANALYSIS
        # ====================================================

        visual_words = [
            "image",
            "figure",
            "diagram",
            "chart",
            "graph",
            "table",
            "picture",
            "visual",
            "shown",
            "illustration"
        ]


        requires_visual_analysis = any(
            word in question.lower()
            for word in visual_words
        )


        if (
            requires_visual_analysis
            and st.session_state.image_paths
        ):

            st.divider()

            st.subheader(
                "👁️ Visual Analysis"
            )

            selected_images = (
                st.session_state.image_paths[:3]
            )


            for image in selected_images:

                try:

                    with st.spinner(
                        f"Analyzing page "
                        f"{image['page']}..."
                    ):

                        visual_answer = (
                            analyze_image(
                                image["path"],
                                question