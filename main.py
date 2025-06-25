import fitz
import os
import requests
import re
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from fpdf import FPDF
from flask import Flask, request as flask_request, Response, jsonify
import threading

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file!")

# --- PDF + FAISS logic ---
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        if page_text:
            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
    doc.close()
    return text

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

class UnicodePDF(FPDF):
    def __init__(self):
        super().__init__()
        font_path = "font/dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf"
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"❌ Font file not found at: {font_path}")
        self.add_font('DejaVu', '', font_path, uni=True)
        self.set_font('DejaVu', '', 12)
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()

    def add_content(self, text):
        for line in text.split('\n'):
            stripped = line.strip()
            if stripped.startswith("#"):
                level = stripped.count("#")
                heading = stripped.replace("#", "").strip()
                self.set_font('DejaVu', 'B', 14 if level == 1 else 12)
                self.multi_cell(0, 10, heading)
                self.ln(2)
            elif stripped.lower().startswith("summary of key points"):
                self.set_font('DejaVu', 'B', 13)
                self.set_text_color(0, 0, 128)
                self.multi_cell(0, 10, stripped)
                self.set_text_color(0, 0, 0)
                self.ln(2)
            elif stripped.lower().startswith("flashcards"):
                self.set_font('DejaVu', 'B', 13)
                self.set_text_color(0, 128, 0)
                self.multi_cell(0, 10, stripped)
                self.set_text_color(0, 0, 0)
                self.ln(2)
            else:
                self.set_font('DejaVu', '', 12)
                self.multi_cell(0, 8, line)
                self.ln(1)

def save_to_pdf(content, question, folder="studyguide"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    safe_question = re.sub(r'[^a-zA-Z0-9_]+', '_', question.strip().lower())
    filename = os.path.join(folder, f"{safe_question}_response.pdf")
    pdf = UnicodePDF()
    pdf.add_content(content)
    pdf.output(filename)
    print(f"✅ Saved response to {filename}")

# --- Embeddings wrapper ---
class LocalEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

    def __call__(self, text):
        return self.embed_query(text)

# --- Build / load FAISS ---
print("🔍 Loading embedding model...")
embedder = LocalEmbeddings()

faiss_index_path = "studyguide/faiss_index"
if os.path.exists(faiss_index_path + ".index"):
    print("✅ Loading existing FAISS index...")
    db = FAISS.load_local(faiss_index_path, embedder)
else:
    print("📄 Processing PDFs + Creating FAISS index...")
    pdf_files = [
        "data/ai book.pdf",
        "data/ml book.pdf",
        "data/fds book.pdf"
    ]
    all_chunks = []
    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            print(f"Processing {pdf_path}...")
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
        else:
            print(f"⚠ File not found: {pdf_path}")
    print(f"✅ Total chunks created: {len(all_chunks)}")
    clean_chunks = [c for c in all_chunks if c.strip()]
    print(f"✅ Clean chunks: {len(clean_chunks)}")
    db = FAISS.from_texts(clean_chunks, embedder)
    db.save_local(faiss_index_path)
    print(f"✅ FAISS index saved at {faiss_index_path}")

# --- Flask app ---
from flask import send_from_directory, Flask, request, jsonify, Response
import threading
import time
import requests

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    # Validate JSON input more robustly
    data = request.get_json(force=True, silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Request JSON must contain 'question'"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    # Perform document similarity search
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

    # Construct prompt
    prompt = f"""
You are an AI tutor generating a study guide. Use the provided context AND your own knowledge.

✅ Structure clearly:
- Main headings / sub-headings
- Definitions, characteristics, applications
- Examples + diagram suggestions (ASCII if possible)
- Clear elaboration

✅ End with:
- Summary of key points (bullets)
- 3 to 5 flashcards (Q&A)

Context:
{context}

Question:
{question}

Answer:
"""

    # Prepare Groq API call
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    groq_payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        groq_response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=groq_payload,
            headers=headers,
            timeout=60
        )
        groq_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Groq API error: {str(e)}"}), 500

    # Extract and return content
    try:
        content = groq_response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return jsonify({"error": "Invalid response structure from Groq API"}), 500

    save_to_pdf(content, question)
    return Response(content, mimetype="text/plain")

@app.route("/", methods=["GET"])
def serve_frontend():
    # Serve your static frontend file
    return send_from_directory("static", "index.html")

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False)

# Start server in a separate thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

time.sleep(2)
print("🚀 Flask server started and ready to receive requests!")

flask_thread.join()

