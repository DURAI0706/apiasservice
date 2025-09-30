import os
import numpy as np
from pymongo import MongoClient
from urllib.parse import quote_plus
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
from datetime import datetime

# ================== Load Environment Variables ==================
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ================== Flask Setup ==================
app = Flask(__name__)
app.secret_key = SECRET_KEY

# ================== MongoDB Connection ==================
encoded_password = quote_plus(MONGO_PASSWORD)
client_mongo = MongoClient(
    f"mongodb+srv://sujan:{encoded_password}@cluster0.6eacggy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
db = client_mongo["apiasservice"]
collection = db["studentdataset"]
teachers_collection = db["teachers"]
logs_collection = db["api_logs"]  # logs collection

# ================== Initialize Teachers Collection ==================
def initialize_teachers():
    try:
        if "teachers" not in db.list_collection_names():
            db.create_collection("teachers")
        
        existing_teacher = teachers_collection.find_one({"username": "sujan"})
        if not existing_teacher:
            teachers_collection.insert_one({
                "username": "sujan",
                "password": "SAWQ#@21",  # In production, hash this!
                "role": "teacher",
                "created_at": "2025-09-29"
            })
            print("✅ Default teacher account created!")
    except Exception as e:
        print(f"Error initializing teachers: {str(e)}")

initialize_teachers()

# ================== Load Data & Build FAISS Index ==================
print("Loading student data and building search index...")
docs = list(collection.find({}, {"_id": 0}))
texts = [str(doc) for doc in docs]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

print(f"✅ Loaded {len(texts)} student records successfully!")

# ================== Groq Client ==================
groq_client = Groq(api_key=GROQ_API_KEY)

def rag_query(user_question, top_k=3):
    try:
        query_embedding = embedder.encode([user_question])
        distances, indices = index.search(np.array(query_embedding), top_k)
        retrieved_docs = [texts[i] for i in indices[0]]

        context = "\n".join(retrieved_docs)
        prompt = f"""
        You are a helpful AI assistant for teachers analyzing student data.
        Context from Student Dataset:
        {context}

        Teacher's Question: {user_question}
        Assistant's Answer:
        """

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=1024,
            top_p=1,
        )

        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in RAG query: {str(e)}")
        return "⚠️ Unable to process your question right now."

# ================== Authentication ==================
def authenticate_teacher(username, password):
    try:
        teacher = teachers_collection.find_one({
            "username": username,
            "password": password
        })
        return teacher is not None
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        return False

# ================== Request Logger ==================

@app.before_request
def log_request():
    log_entry = {
        "time": datetime.now(timezone.utc),  # timezone-aware datetime
        "method": request.method,
        "path": request.path,
        "ip": request.remote_addr,
        "user": session.get("user", "anonymous")
    }
    logs_collection.insert_one(log_entry)
    print(f"[API HIT] {log_entry}")


# ================== Routes ==================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if authenticate_teacher(username, password):
            session["user"] = username
            return redirect(url_for("chatbot"))
        else:
            return render_template("login.html", error="❌ Invalid username or password.")
    return render_template("login.html")

@app.route("/chatbot")
def chatbot():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("chatbot.html")

@app.route("/ask", methods=["POST"])
def ask():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_question = request.json.get("question", "").strip()
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = rag_query(user_question)
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return jsonify({"error": "Processing error"}), 500

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "total_students": len(texts)})

@app.route("/logs")
def view_logs():
    if "user" not in session:
        return redirect(url_for("login"))

    logs = list(logs_collection.find().sort("time", -1).limit(50))
    return render_template("logs.html", logs=logs)

# ================== Error Handlers ==================
@app.errorhandler(404)
def not_found(error):
    return redirect(url_for("login"))

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ================== Main ==================
if __name__ == "__main__":
    print("🚀 Starting Teacher Portal...")
    print("Login credentials:")
    print("Username: sujan")
    print("Password: SAWQ#@21")
    print("-" * 40)
    app.run(debug=True, host="0.0.0.0", port=5000)
