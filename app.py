import os
import json
import re
import uuid
import pickle
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from werkzeug.utils import secure_filename
from datetime import datetime
from io import BytesIO
from docx import Document

# ================================================================
# APP CONFIGURATION
# ================================================================
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nigat.db'
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# FAISS storage directory
FAISS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faiss_indexes')
if not os.path.exists(FAISS_DIR):
    os.makedirs(FAISS_DIR)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ================================================================
# DATABASE INITIALIZATION
# ================================================================
db = SQLAlchemy(app)

# ================================================================
# OPENROUTER CONFIGURATION
# ================================================================

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free models with fallback support
FREE_MODELS = [
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
]

current_model_index = 0
model_failures = {}

def get_next_model():
    global current_model_index
    available_models = [m for m in FREE_MODELS if m not in model_failures]
    if not available_models:
        model_failures.clear()
        available_models = FREE_MODELS
    
    model = available_models[current_model_index % len(available_models)]
    current_model_index += 1
    return model

def get_ai_response(system_prompt, user_query, context_chunks):
    if not OPENROUTER_API_KEY:
        return "⚠️ OpenRouter API key is not set. Please add OPENROUTER_API_KEY to environment variables."
    
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context available."
    prompt_type = detect_prompt_type(user_query)
    prompt_template = get_prompt_template(prompt_type)
    
    if detect_language(user_query) == 'amharic':
        lang_instruction = "You MUST respond in Amharic (በአማርኛ)."
    else:
        lang_instruction = "You MUST respond in English."
    
    full_prompt = (
        f"{system_prompt}\n\n"
        f"=== LANGUAGE ===\n{lang_instruction}\n\n"
        f"=== TASK ===\n{prompt_template}\n\n"
        f"=== CONTEXT ===\n{context_text}\n\n"
        f"=== USER QUESTION ===\n{user_query}"
    )
    
    estimated_tokens = len(full_prompt) // 4
    print(f"📊 Estimated tokens: {estimated_tokens}")
    
    max_attempts = len(FREE_MODELS) * 2
    for attempt in range(max_attempts):
        model = get_next_model()
        print(f"🤖 Attempt {attempt+1}: Using {model}")
        
        try:
            response = requests.post(
                OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://nigat-tutor-ai.onrender.com",
                    "X-Title": "Nigat Tutor AI"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt}
                    ],
                    "temperature": 0.05,
                    "max_tokens": 1024,
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and data['choices']:
                    print(f"✅ Response received from {model}")
                    if model in model_failures:
                        del model_failures[model]
                    return data['choices'][0]['message']['content']
            else:
                error_msg = response.text[:200] if response.text else "Unknown error"
                print(f"⚠️ {model} error: {response.status_code} - {error_msg}")
                model_failures[model] = True
                if response.status_code == 429:
                    print(f"⏳ Rate limit on {model}, switching to next model...")
                    continue
                    
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout on {model}, trying next...")
            model_failures[model] = True
            continue
        except Exception as e:
            print(f"⚠️ Error with {model}: {e}")
            model_failures[model] = True
            continue
    
    return "⚠️ All available models failed. Please try again later or check your OpenRouter API key."

# ================================================================
# PROMPT MANAGEMENT
# ================================================================

def detect_prompt_type(query):
    query_lower = query.lower()
    if any(w in query_lower for w in ['daily lesson', 'ዕለታዊ', 'lesson plan']):
        return 'daily_lesson'
    elif any(w in query_lower for w in ['annual', 'ዓመታዊ', 'yearly']):
        return 'annual_plan'
    elif any(w in query_lower for w in ['exam', 'test', 'quiz', 'ፈተና', 'ምዘና']):
        return 'exam'
    elif any(w in query_lower for w in ['summary', 'summarize', 'ማጠቃለያ']):
        return 'summary'
    return 'general'

def get_prompt_template(prompt_type):
    templates = {
        'daily_lesson': """
=== DAILY LESSON PLAN ===
Generate a complete daily lesson plan with this structure:

1. SCHOOL INFORMATION: School Name, Teacher Name, Grade/Section, Subject, Date, Unit, Topic, Page
2. LESSON OVERVIEW: Rationale, Prerequisites, Competencies (3-5 bullet points)
3. LESSON STAGES: A table with: Stage | Time | Teacher Activities | Student Activities | Methodology | Assessment
4. SUPPORT FOR LEARNERS: Table with: Category (Slow/Medium/Fast) | Support Strategies
5. APPROVALS: Table with: Role | Name | Signature | Date
6. TEACHER'S SELF-ASSESSMENT: Brief reflection

Use information from the provided context to fill in the content.
""",
        'annual_plan': """
=== ANNUAL LESSON PLAN ===
Generate an annual lesson plan with this structure:

1. SCHOOL INFORMATION: School Name, Teacher Name, Subject, Grade, Academic Year
2. TABLE: Month | Week | Topics | Objectives | Methodology | Evaluation

Use information from the provided context.
""",
        'exam': """
=== EXAM GENERATOR ===
Generate exam questions based on the provided content.
Include:
- Multiple choice questions (4 options each)
- True/False questions
- Short answer questions

Use the context to create accurate, relevant questions.
""",
        'summary': """
=== SUMMARY GENERATOR ===
Create a comprehensive summary of the provided content.
Include:
- Main topics and key points
- Important concepts
- Key terms and definitions

Keep the summary well-structured and easy to read.
""",
        'general': """
=== GENERAL ASSISTANCE ===
Answer the user's question based on the provided context.
Be helpful, accurate, and cite specific information from the context.
If the context doesn't contain the answer, say so clearly.
"""
    }
    return templates.get(prompt_type, templates['general'])

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def detect_language(text):
    if not text:
        return 'english'
    amharic_pattern = re.compile(r'[\u1200-\u137F]')
    if amharic_pattern.search(text):
        return 'amharic'
    return 'english'

def remove_duplicate_sentences(text):
    if not text:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique = []
    for sentence in sentences:
        s = sentence.strip()
        if not s or len(s) < 5:
            continue
        norm = s.lower()
        if norm not in seen:
            seen.add(norm)
            unique.append(s)
    return ' '.join(unique)

# ================================================================
# MEMORY OPTIMIZED PDF EXTRACTION
# ================================================================

def extract_pdf_text_streaming(filepath):
    """Extract PDF text page by page with memory optimization"""
    try:
        import pdfplumber
        text_parts = []
        total_pages = 0
        
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            # Process only first 50 pages to save memory
            max_pages = min(total_pages, 50)
            
            for i in range(max_pages):
                page = pdf.pages[i]
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                # Free page memory
                page = None
                if (i + 1) % 10 == 0:
                    print(f"📄 Extracted page {i+1}/{max_pages}")
        
        full_text = "\n\n".join(text_parts)
        # Free text_parts memory
        text_parts = None
        return full_text, max_pages if full_text else "No text found in PDF.", 0
    except Exception as e:
        return f"PDF extraction error: {str(e)}", 0

# ================================================================
# EMBEDDING MODEL (Singleton)
# ================================================================

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("🔄 Loading embedding model...")
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded")
        except ImportError:
            print("⚠️ sentence-transformers not installed!")
            _embedding_model = None
    return _embedding_model

def get_embedding(text):
    model = get_embedding_model()
    if model is None:
        import hashlib
        words = text.lower().split()
        vector = np.zeros(384)
        for word in words[:100]:
            h = hashlib.md5(word.encode()).hexdigest()
            for i in range(min(8, len(h))):
                vector[i % 384] += (int(h[i], 16) - 8) / 16
        norm = np.linalg.norm(vector)
        return vector / (norm + 1e-8)
    try:
        return model.encode(text, normalize_embeddings=True)
    except:
        return np.zeros(384)

# ================================================================
# MEMORY OPTIMIZED RAG SYSTEM
# ================================================================

class EnterpriseRAG:
    def __init__(self):
        self.doc_metadata = {}
        self.chunk_texts = {}
        self.faiss_indexes = {}
        self.chunk_size = 300  # ቀንሷል
        self.overlap = 50      # ቀንሷል
        self.max_chunks = 500  # ከፍተኛ የክፍል ብዛት
    
    def get_index_path(self, session_id):
        return os.path.join(FAISS_DIR, f"{session_id}.faiss")
    
    def get_metadata_path(self, session_id):
        return os.path.join(FAISS_DIR, f"{session_id}_meta.pkl")
    
    def _chunk_text_streaming(self, text):
        words = text.split()
        total_words = len(words)
        chunk_count = 0
        for start in range(0, total_words, self.chunk_size - self.overlap):
            if chunk_count >= self.max_chunks:
                break
            end = min(start + self.chunk_size, total_words)
            chunk_words = words[start:end]
            if len(chunk_words) < 15:
                continue
            chunk_count += 1
            yield ' '.join(chunk_words)
            if end >= total_words:
                break
    
    def store_document(self, session_id, text, filename, pages=0):
        self.doc_metadata[session_id] = {
            'filename': filename,
            'pages': pages,
            'word_count': len(text.split()),
            'chunk_count': 0
        }
        
        try:
            import faiss
            embedding_dim = 384
            faiss_index = faiss.IndexFlatIP(embedding_dim)
        except ImportError:
            faiss_index = None
        
        chunks = []
        embeddings = []
        chunk_count = 0
        
        for chunk_text in self._chunk_text_streaming(text):
            chunks.append(chunk_text)
            emb = get_embedding(chunk_text)
            embeddings.append(emb)
            chunk_count += 1
            
            # Add to FAISS in smaller batches (50 instead of 100)
            if faiss_index is not None and len(embeddings) >= 50:
                if embeddings:
                    emb_array = np.array(embeddings).astype('float32')
                    faiss_index.add(emb_array)
                    embeddings = []
                    # Force garbage collection
                    import gc
                    gc.collect()
        
        # Add remaining embeddings
        if embeddings and faiss_index is not None:
            emb_array = np.array(embeddings).astype('float32')
            faiss_index.add(emb_array)
            embeddings = None
        
        self.chunk_texts[session_id] = chunks
        self.doc_metadata[session_id]['chunk_count'] = len(chunks)
        
        if faiss_index is not None:
            faiss_path = self.get_index_path(session_id)
            faiss.write_index(faiss_index, faiss_path)
            self.faiss_indexes[session_id] = faiss_index
        
        meta_path = self.get_metadata_path(session_id)
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'chunks': chunks,
                'metadata': self.doc_metadata[session_id]
            }, f)
        
        print(f"📚 Stored {len(chunks)} chunks")
        return len(chunks)
    
    def _load_faiss_index(self, session_id):
        if session_id in self.faiss_indexes:
            return self.faiss_indexes[session_id]
        try:
            import faiss
            faiss_path = self.get_index_path(session_id)
            if os.path.exists(faiss_path):
                index = faiss.read_index(faiss_path)
                self.faiss_indexes[session_id] = index
                return index
        except:
            pass
        return None
    
    def _load_metadata(self, session_id):
        meta_path = self.get_metadata_path(session_id)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'rb') as f:
                    data = pickle.load(f)
                    self.chunk_texts[session_id] = data.get('chunks', [])
                    self.doc_metadata[session_id] = data.get('metadata', {})
                    return data
            except:
                pass
        return None
    
    def get_relevant_chunks(self, session_id, query, max_tokens=4000):
        if session_id not in self.chunk_texts:
            self._load_metadata(session_id)
        
        if session_id not in self.chunk_texts or not self.chunk_texts[session_id]:
            return []
        
        faiss_index = self._load_faiss_index(session_id)
        chunks = self.chunk_texts[session_id]
        
        if faiss_index is not None:
            try:
                import faiss
                query_emb = get_embedding(query)
                query_emb = np.array([query_emb]).astype('float32')
                
                k = min(20, len(chunks))
                scores, indices = faiss_index.search(query_emb, k)
                
                selected = []
                total_tokens = 0
                for idx in indices[0]:
                    if idx < 0 or idx >= len(chunks):
                        continue
                    chunk = chunks[idx]
                    estimated = len(chunk) // 4
                    if total_tokens + estimated <= max_tokens:
                        selected.append(chunk)
                        total_tokens += estimated
                    if len(selected) >= 5:  # Limit to 5 chunks max
                        break
                return selected if selected else [chunks[0]]
            except:
                pass
        
        # Keyword fallback
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        scored = []
        for i, chunk in enumerate(chunks[:100]):  # Only check first 100 chunks
            chunk_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', chunk.lower()))
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                scored.append((overlap, chunk))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:3]]
    
    def get_document_info(self, session_id):
        if session_id in self.doc_metadata:
            return self.doc_metadata[session_id]
        self._load_metadata(session_id)
        return self.doc_metadata.get(session_id)
    
    def clear(self, session_id):
        if session_id in self.doc_metadata:
            del self.doc_metadata[session_id]
        if session_id in self.chunk_texts:
            del self.chunk_texts[session_id]
        if session_id in self.faiss_indexes:
            del self.faiss_indexes[session_id]
        
        faiss_path = self.get_index_path(session_id)
        if os.path.exists(faiss_path):
            os.remove(faiss_path)
        meta_path = self.get_metadata_path(session_id)
        if os.path.exists(meta_path):
            os.remove(meta_path)

rag = EnterpriseRAG()

# ================================================================
# MODELS
# ================================================================
class Course(db.Model):
    __tablename__ = 'course'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    color = db.Column(db.String(20), default='blue')
    quiz_link = db.Column(db.String(500))

class AnnualPlan(db.Model):
    __tablename__ = 'annual_plan'
    id = db.Column(db.Integer, primary_key=True)
    school_name = db.Column(db.String(200))
    teacher_name = db.Column(db.String(100))
    subject = db.Column(db.String(100))
    grade = db.Column(db.Integer)
    section = db.Column(db.String(20))
    year = db.Column(db.Integer)
    total_days = db.Column(db.Integer)
    unit_number = db.Column(db.Integer)
    unit_title = db.Column(db.String(200))
    unit_objectives = db.Column(db.Text)
    month_data = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class LaboratoryPlan(db.Model):
    __tablename__ = 'laboratory_plan'
    id = db.Column(db.Integer, primary_key=True)
    school_name = db.Column(db.String(200))
    teacher_name = db.Column(db.String(100))
    subject = db.Column(db.String(100))
    grade = db.Column(db.Integer)
    year = db.Column(db.Integer)
    experiment_data = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class DailyPlan(db.Model):
    __tablename__ = 'daily_plan'
    id = db.Column(db.Integer, primary_key=True)
    teacher_name = db.Column(db.String(100))
    school_name = db.Column(db.String(200))
    grade = db.Column(db.Integer)
    section = db.Column(db.String(20))
    subject = db.Column(db.String(100))
    date = db.Column(db.Date)
    unit_number = db.Column(db.Integer)
    lesson_topic = db.Column(db.String(200))
    page = db.Column(db.String(20))
    rationale = db.Column(db.Text)
    prerequisites = db.Column(db.Text)
    competencies = db.Column(db.Text)
    starter_time = db.Column(db.Integer)
    starter_teacher = db.Column(db.Text)
    starter_student = db.Column(db.Text)
    starter_method = db.Column(db.String(100))
    starter_assessment = db.Column(db.String(100))
    starter_aids = db.Column(db.String(200))
    main_time = db.Column(db.Integer)
    main_teacher = db.Column(db.Text)
    main_student = db.Column(db.Text)
    main_method = db.Column(db.String(100))
    main_assessment = db.Column(db.String(100))
    main_aids = db.Column(db.String(200))
    conclude_time = db.Column(db.Integer)
    conclude_teacher = db.Column(db.Text)
    conclude_student = db.Column(db.Text)
    conclude_method = db.Column(db.String(100))
    conclude_assessment = db.Column(db.String(100))
    conclude_aids = db.Column(db.String(200))
    slow_learners = db.Column(db.Text)
    medium_learners = db.Column(db.Text)
    fast_learners = db.Column(db.Text)
    self_assessment = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PeaceClubPlan(db.Model):
    __tablename__ = 'peace_club_plan'
    id = db.Column(db.Integer, primary_key=True)
    school_name = db.Column(db.String(200))
    district = db.Column(db.String(200))
    woreda = db.Column(db.String(100))
    school_level = db.Column(db.String(200))
    club_name = db.Column(db.String(200), default='Peace Club / የሰላም ክበብ')
    teacher_name = db.Column(db.String(100))
    teacher_signature = db.Column(db.String(100))
    secretary_name = db.Column(db.String(100))
    secretary_signature = db.Column(db.String(100))
    year = db.Column(db.Integer)
    month = db.Column(db.String(20))
    vision = db.Column(db.Text)
    mission = db.Column(db.Text)
    opportunities = db.Column(db.Text)
    challenges = db.Column(db.Text)
    solutions = db.Column(db.Text)
    action_plan = db.Column(db.Text)
    student_members = db.Column(db.Text)
    teacher_members = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_action_plan(self):
        return json.loads(self.action_plan) if self.action_plan else []
    
    def get_student_members(self):
        return json.loads(self.student_members) if self.student_members else []
    
    def get_teacher_members(self):
        return json.loads(self.teacher_members) if self.teacher_members else []

class PeaceClubActivity(db.Model):
    __tablename__ = 'peace_club_activity'
    id = db.Column(db.Integer, primary_key=True)
    club_plan_id = db.Column(db.Integer, db.ForeignKey('peace_club_plan.id'))
    activity_number = db.Column(db.Integer)
    activity_name = db.Column(db.String(500))
    hamle = db.Column(db.Boolean, default=False)
    nehase = db.Column(db.Boolean, default=False)
    meskerem = db.Column(db.Boolean, default=False)
    tikimt = db.Column(db.Boolean, default=False)
    hidar = db.Column(db.Boolean, default=False)
    tahsas = db.Column(db.Boolean, default=False)
    tir = db.Column(db.Boolean, default=False)
    yekatit = db.Column(db.Boolean, default=False)
    megabit = db.Column(db.Boolean, default=False)
    miazia = db.Column(db.Boolean, default=False)
    ginbot = db.Column(db.Boolean, default=False)
    sene = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

admin = Admin(app, name='Nigat Admin')
admin.add_view(ModelView(Course, db))
admin.add_view(ModelView(AnnualPlan, db))
admin.add_view(ModelView(LaboratoryPlan, db))
admin.add_view(ModelView(DailyPlan, db))
admin.add_view(ModelView(PeaceClubPlan, db))
admin.add_view(ModelView(PeaceClubActivity, db))

# ================================================================
# ROUTES
# ================================================================
@app.route('/')
def home():
    try:
        courses = Course.query.all()
        return render_template('index.html', courses=courses)
    except Exception as e:
        print(f"❌ Home route error: {e}")
        with app.app_context():
            db.create_all()
        courses = Course.query.all()
        return render_template('index.html', courses=courses)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        flash('Registration submitted successfully!', 'success')
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/course/<int:course_id>')
def course_detail(course_id):
    return render_template('course_detail.html', course=Course.query.get_or_404(course_id))

# ================================================================
# UPLOAD ROUTES (Memory Optimized)
# ================================================================
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected.'}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            
            # Limit to 50 pages max for memory
            text, pages = extract_pdf_text_streaming(filepath)
            
            if not text or text.startswith("PDF extraction error"):
                return jsonify({'success': False, 'message': f'Error extracting text: {text}'}), 500
            
            if 'session_id' not in session:
                session['session_id'] = str(uuid.uuid4())
            
            session_id = session['session_id']
            num_chunks = rag.store_document(session_id, text, filename, pages)
            
            session['pdf_filename'] = filename
            session['pdf_size'] = file_size
            session['pdf_pages'] = pages
            session['pdf_chunks'] = num_chunks
            
            return jsonify({
                'success': True, 
                'message': f'PDF uploaded and indexed! ({file_size:.1f}MB, {pages} pages, {num_chunks} chunks)',
                'session_id': session_id,
                'pages': pages,
                'chunks': num_chunks
            }), 200
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500
    else:
        return jsonify({'success': False, 'message': 'Only PDF files are allowed.'}), 400

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'message': 'Image uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

@app.route('/clear_context', methods=['POST'])
def clear_context():
    session_id = session.get('session_id')
    if session_id:
        rag.clear(session_id)
    session.pop('pdf_filename', None)
    session.pop('pdf_size', None)
    session.pop('pdf_pages', None)
    session.pop('pdf_chunks', None)
    return jsonify({'message': 'Context cleared successfully'}), 200

# ================================================================
# AI CHAT ROUTE
# ================================================================
@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    user_query = request.json.get('query', '').strip()
    
    if not user_query:
        return jsonify({"answer": "Please ask a question."})
    
    query_lang = detect_language(user_query)
    print(f"🔍 Detected language: {query_lang}")
    print(f"📝 User query: {user_query[:100]}...")
    
    session_id = session.get('session_id')
    
    if not session_id:
        return jsonify({"answer": "Please upload a document first before asking questions."})
    
    relevant_chunks = rag.get_relevant_chunks(session_id, user_query, max_tokens=4000)
    
    if not relevant_chunks:
        return jsonify({"answer": "I couldn't find relevant information in the uploaded document. Please try a different question."})
    
    doc_info = rag.get_document_info(session_id)
    if doc_info:
        print(f"📄 Document: {doc_info.get('filename', 'unknown')} ({doc_info.get('pages', 0)} pages, {doc_info.get('chunk_count', 0)} chunks)")
    
    print(f"📚 Retrieved {len(relevant_chunks)} relevant chunks")
    
    if query_lang == 'amharic':
        language_instruction = "You MUST respond in Amharic (በአማርኛ)."
    else:
        language_instruction = "You MUST respond in English."
    
    system_prompt = (
        "You are 'Nigat AI Tutor'. Created by Teacher Fisaha Melke.\n\n"
        f"=== LANGUAGE RULE ===\n{language_instruction}\n"
        "Do NOT switch languages.\n\n"
        "=== ABOUT THE CREATOR ===\n"
        "My name is Fisiha Melke. I graduated from Ambo University with a Bachelor's degree in Biology in 2024 (2016 E.C.). I have more than two years of teaching experience in private schools. I hold a Certificate in Video Editing. I am currently developing Nigat Tutor AI, an educational platform designed to support Ethiopian teachers and students.\n\n"
        "=== ACCURACY RULE ===\n"
        "Provide ONLY accurate information based on the provided context. If the context doesn't contain the answer, say so clearly.\n\n"
        "=== AMHARIC SPELLING ===\n"
        "Correct spellings: 'ጎንደር' (not ንንደር/ጀንደር), 'ኢትዮጵያ' (not እትዮጵያ).\n"
    )
    
    answer = get_ai_response(system_prompt, user_query, relevant_chunks)
    
    if answer:
        answer = remove_duplicate_sentences(answer)
    
    return jsonify({"answer": answer or "⚠️ No AI response available. Please try again later."})

# ================================================================
# DOWNLOAD WORD
# ================================================================
@app.route('/download_word', methods=['POST'])
def download_word():
    data = request.json
    content = data.get('content', '')
    filename = data.get('filename', 'Nigat_AI_Response')
    
    if not content:
        return jsonify({'error': 'No content to download'}), 400
    
    try:
        doc = Document()
        doc.add_heading('Nigat AI Tutor Response', 0)
        
        lines = content.split('\n')
        in_table = False
        table_rows = []
        table_headers = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('|') and line.endswith('|'):
                cells = [cell.strip() for cell in line[1:-1].split('|')]
                
                if all('---' in cell or ':' in cell for cell in cells):
                    continue
                
                if not in_table:
                    in_table = True
                    table_headers = cells
                else:
                    table_rows.append(cells)
            else:
                if in_table and table_rows:
                    num_cols = max(len(table_headers), max([len(row) for row in table_rows]) if table_rows else 0)
                    
                    if num_cols > 0 and table_headers:
                        table = doc.add_table(rows=1 + len(table_rows), cols=num_cols)
                        table.style = 'Table Grid'
                        
                        for i, header in enumerate(table_headers[:num_cols]):
                            cell = table.cell(0, i)
                            cell.text = header
                            for paragraph in cell.paragraphs:
                                for run in paragraph.runs:
                                    run.bold = True
                        
                        for row_idx, row in enumerate(table_rows):
                            for col_idx, cell_text in enumerate(row[:num_cols]):
                                table.cell(row_idx + 1, col_idx).text = cell_text
                    
                    table_rows = []
                    table_headers = []
                    in_table = False
                
                if line:
                    if line.startswith('#'):
                        heading_level = min(len(line) - len(line.lstrip('#')), 6)
                        heading_text = line.lstrip('#').strip()
                        doc.add_heading(heading_text, level=heading_level)
                    else:
                        doc.add_paragraph(line)
        
        if in_table and table_rows:
            num_cols = max(len(table_headers), max([len(row) for row in table_rows]) if table_rows else 0)
            if num_cols > 0 and table_headers:
                table = doc.add_table(rows=1 + len(table_rows), cols=num_cols)
                table.style = 'Table Grid'
                
                for i, header in enumerate(table_headers[:num_cols]):
                    cell = table.cell(0, i)
                    cell.text = header
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.bold = True
                
                for row_idx, row in enumerate(table_rows):
                    for col_idx, cell_text in enumerate(row[:num_cols]):
                        table.cell(row_idx + 1, col_idx).text = cell_text
        
        file_stream = BytesIO()
        doc.save(file_stream)
        file_stream.seek(0)
        
        safe_filename = re.sub(r'[^\w\s-]', '', filename)
        safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
        
        return send_file(
            file_stream,
            as_attachment=True,
            download_name=f"{safe_filename}.docx",
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
    except Exception as e:
        print(f"❌ Word download error: {e}")
        return jsonify({'error': f'Failed to generate document: {str(e)}'}), 500

# ================================================================
# LESSON PLAN ROUTES
# ================================================================
@app.route('/lesson')
def lesson_home():
    annual_plans = AnnualPlan.query.all()
    laboratory_plans = LaboratoryPlan.query.all()
    daily_plans = DailyPlan.query.all()
    return render_template('lesson_plan.html', 
                         annual_plans=annual_plans,
                         laboratory_plans=laboratory_plans,
                         daily_plans=daily_plans)

@app.route('/lesson/annual', methods=['GET', 'POST'])
def annual_plan():
    if request.method == 'POST':
        months = request.form.getlist('month[]')
        weeks = request.form.getlist('week[]')
        periods = request.form.getlist('period[]')
        date_ranges = request.form.getlist('date_range[]')
        pages = request.form.getlist('page[]')
        topics = request.form.getlist('topics[]')
        objectives = request.form.getlist('objectives[]')
        methodologies = request.form.getlist('methodology[]')
        teaching_aids = request.form.getlist('teaching_aids[]')
        evaluations = request.form.getlist('evaluation[]')
        
        month_data = []
        for i in range(len(months)):
            month_data.append({
                'month': months[i],
                'week': weeks[i],
                'period': periods[i],
                'date_range': date_ranges[i],
                'page': pages[i],
                'topics': topics[i],
                'objectives': objectives[i],
                'methodology': methodologies[i],
                'teaching_aids': teaching_aids[i],
                'evaluation': evaluations[i]
            })
        
        plan = AnnualPlan(
            school_name=request.form.get('school_name'),
            teacher_name=request.form.get('teacher_name'),
            subject=request.form.get('subject'),
            grade=int(request.form.get('grade')) if request.form.get('grade') else None,
            section=request.form.get('section'),
            year=int(request.form.get('year')) if request.form.get('year') else None,
            total_days=int(request.form.get('total_days')) if request.form.get('total_days') else None,
            unit_number=int(request.form.get('unit_number')) if request.form.get('unit_number') else None,
            unit_title=request.form.get('unit_title'),
            unit_objectives=request.form.get('unit_objectives'),
            month_data=str(month_data)
        )
        db.session.add(plan)
        db.session.commit()
        flash('Annual plan created successfully!', 'success')
        return redirect(url_for('lesson_home'))
    
    return render_template('annual_plan_form.html')

@app.route('/lesson/laboratory', methods=['GET', 'POST'])
def laboratory_plan():
    if request.method == 'POST':
        exp_numbers = request.form.getlist('exp_number[]')
        exp_titles = request.form.getlist('exp_title[]')
        apparatus_list = request.form.getlist('apparatus[]')
        chemicals_list = request.form.getlist('chemicals[]')
        unit_numbers = request.form.getlist('unit_number[]')
        pages = request.form.getlist('page[]')
        months = request.form.getlist('month[]')
        dates = request.form.getlist('date[]')
        
        experiment_data = []
        for i in range(len(exp_numbers)):
            experiment_data.append({
                'exp_number': exp_numbers[i],
                'exp_title': exp_titles[i],
                'apparatus': apparatus_list[i],
                'chemicals': chemicals_list[i],
                'unit_number': unit_numbers[i],
                'page': pages[i],
                'month': months[i],
                'date': dates[i]
            })
        
        plan = LaboratoryPlan(
            school_name=request.form.get('school_name'),
            teacher_name=request.form.get('teacher_name'),
            subject=request.form.get('subject'),
            grade=int(request.form.get('grade')) if request.form.get('grade') else None,
            year=int(request.form.get('year')) if request.form.get('year') else None,
            experiment_data=str(experiment_data)
        )
        db.session.add(plan)
        db.session.commit()
        flash('Laboratory plan created successfully!', 'success')
        return redirect(url_for('lesson_home'))
    
    return render_template('laboratory_plan_form.html')

@app.route('/lesson/daily', methods=['GET', 'POST'])
def daily_plan():
    if request.method == 'POST':
        date_str = request.form.get('date')
        plan = DailyPlan(
            teacher_name=request.form.get('teacher_name'),
            school_name=request.form.get('school_name'),
            grade=int(request.form.get('grade')) if request.form.get('grade') else None,
            section=request.form.get('section'),
            subject=request.form.get('subject'),
            date=datetime.strptime(date_str, '%Y-%m-%d').date() if date_str else None,
            unit_number=int(request.form.get('unit_number')) if request.form.get('unit_number') else None,
            lesson_topic=request.form.get('lesson_topic'),
            page=request.form.get('page'),
            rationale=request.form.get('rationale'),
            prerequisites=request.form.get('prerequisites'),
            competencies=request.form.get('competencies'),
            starter_time=int(request.form.get('starter_time')) if request.form.get('starter_time') else None,
            starter_teacher=request.form.get('starter_teacher'),
            starter_student=request.form.get('starter_student'),
            starter_method=request.form.get('starter_method'),
            starter_assessment=request.form.get('starter_assessment'),
            starter_aids=request.form.get('starter_aids'),
            main_time=int(request.form.get('main_time')) if request.form.get('main_time') else None,
            main_teacher=request.form.get('main_teacher'),
            main_student=request.form.get('main_student'),
            main_method=request.form.get('main_method'),
            main_assessment=request.form.get('main_assessment'),
            main_aids=request.form.get('main_aids'),
            conclude_time=int(request.form.get('conclude_time')) if request.form.get('conclude_time') else None,
            conclude_teacher=request.form.get('conclude_teacher'),
            conclude_student=request.form.get('conclude_student'),
            conclude_method=request.form.get('conclude_method'),
            conclude_assessment=request.form.get('conclude_assessment'),
            conclude_aids=request.form.get('conclude_aids'),
            slow_learners=request.form.get('slow_learners'),
            medium_learners=request.form.get('medium_learners'),
            fast_learners=request.form.get('fast_learners'),
            self_assessment=request.form.get('self_assessment')
        )
        db.session.add(plan)
        db.session.commit()
        flash('Daily plan created successfully!', 'success')
        return redirect(url_for('lesson_home'))
    
    return render_template('daily_plan_form.html')

# ================================================================
# PEACE CLUB ROUTES
# ================================================================
@app.route('/peaceclub')
def peaceclub_home():
    club_plans = PeaceClubPlan.query.all()
    return render_template('peaceclub_home.html', club_plans=club_plans)

@app.route('/peaceclub/create', methods=['GET', 'POST'])
def peaceclub_create():
    if request.method == 'POST':
        plan = PeaceClubPlan(
            school_name=request.form.get('school_name'),
            district=request.form.get('district'),
            woreda=request.form.get('woreda'),
            school_level=request.form.get('school_level'),
            club_name=request.form.get('club_name', 'Peace Club / የሰላም ክበብ'),
            teacher_name=request.form.get('teacher_name'),
            teacher_signature=request.form.get('teacher_signature'),
            secretary_name=request.form.get('secretary_name'),
            secretary_signature=request.form.get('secretary_signature'),
            year=int(request.form.get('year')) if request.form.get('year') else None,
            month=request.form.get('month'),
            vision=request.form.get('vision'),
            mission=request.form.get('mission'),
            opportunities=request.form.get('opportunities'),
            challenges=request.form.get('challenges'),
            solutions=request.form.get('solutions')
        )
        db.session.add(plan)
        db.session.flush()
        
        activity_names = request.form.getlist('activity_name[]')
        hamle_values = request.form.getlist('hamle')
        nehase_values = request.form.getlist('nehase')
        meskerem_values = request.form.getlist('meskerem')
        tikimt_values = request.form.getlist('tikimt')
        hidar_values = request.form.getlist('hidar')
        tahsas_values = request.form.getlist('tahsas')
        tir_values = request.form.getlist('tir')
        yekatit_values = request.form.getlist('yekatit')
        megabit_values = request.form.getlist('megabit')
        miazia_values = request.form.getlist('miazia')
        ginbot_values = request.form.getlist('ginbot')
        sene_values = request.form.getlist('sene')
        
        for i, name in enumerate(activity_names):
            if name.strip():
                activity = PeaceClubActivity(
                    club_plan_id=plan.id,
                    activity_number=i + 1,
                    activity_name=name.strip(),
                    hamle=str(i) in hamle_values,
                    nehase=str(i) in nehase_values,
                    meskerem=str(i) in meskerem_values,
                    tikimt=str(i) in tikimt_values,
                    hidar=str(i) in hidar_values,
                    tahsas=str(i) in tahsas_values,
                    tir=str(i) in tir_values,
                    yekatit=str(i) in yekatit_values,
                    megabit=str(i) in megabit_values,
                    miazia=str(i) in miazia_values,
                    ginbot=str(i) in ginbot_values,
                    sene=str(i) in sene_values
                )
                db.session.add(activity)
        
        student_names = request.form.getlist('student_name[]')
        student_grades = request.form.getlist('student_grade[]')
        student_data = []
        for i in range(len(student_names)):
            if student_names[i].strip():
                student_data.append({
                    'name': student_names[i].strip(),
                    'grade': student_grades[i] if i < len(student_grades) else ''
                })
        plan.student_members = json.dumps(student_data)
        
        teacher_names = request.form.getlist('teacher_name[]')
        teacher_grades = request.form.getlist('teacher_grade[]')
        teacher_data = []
        for i in range(len(teacher_names)):
            if teacher_names[i].strip():
                teacher_data.append({
                    'name': teacher_names[i].strip(),
                    'grade': teacher_grades[i] if i < len(teacher_grades) else ''
                })
        plan.teacher_members = json.dumps(teacher_data)
        
        db.session.commit()
        flash('Peace Club plan created successfully!', 'success')
        return redirect(url_for('peaceclub_home'))
    
    return render_template('peaceclub_create.html')

@app.route('/peaceclub/view/<int:plan_id>')
def peaceclub_view(plan_id):
    plan = PeaceClubPlan.query.get_or_404(plan_id)
    activities = PeaceClubActivity.query.filter_by(club_plan_id=plan_id).order_by(PeaceClubActivity.activity_number).all()
    student_members = plan.get_student_members()
    teacher_members = plan.get_teacher_members()
    return render_template('peaceclub_view.html', 
                         plan=plan, 
                         activities=activities,
                         student_members=student_members,
                         teacher_members=teacher_members)

@app.route('/peaceclub/edit/<int:plan_id>', methods=['GET', 'POST'])
def peaceclub_edit(plan_id):
    plan = PeaceClubPlan.query.get_or_404(plan_id)
    activities = PeaceClubActivity.query.filter_by(club_plan_id=plan_id).order_by(PeaceClubActivity.activity_number).all()
    student_members = plan.get_student_members()
    teacher_members = plan.get_teacher_members()
    
    if request.method == 'POST':
        plan.school_name = request.form.get('school_name')
        plan.district = request.form.get('district')
        plan.woreda = request.form.get('woreda')
        plan.school_level = request.form.get('school_level')
        plan.club_name = request.form.get('club_name', 'Peace Club / የሰላም ክበብ')
        plan.teacher_name = request.form.get('teacher_name')
        plan.teacher_signature = request.form.get('teacher_signature')
        plan.secretary_name = request.form.get('secretary_name')
        plan.secretary_signature = request.form.get('secretary_signature')
        plan.year = int(request.form.get('year')) if request.form.get('year') else None
        plan.month = request.form.get('month')
        plan.vision = request.form.get('vision')
        plan.mission = request.form.get('mission')
        plan.opportunities = request.form.get('opportunities')
        plan.challenges = request.form.get('challenges')
        plan.solutions = request.form.get('solutions')
        
        for activity in activities:
            db.session.delete(activity)
        
        activity_names = request.form.getlist('activity_name[]')
        hamle_values = request.form.getlist('hamle')
        nehase_values = request.form.getlist('nehase')
        meskerem_values = request.form.getlist('meskerem')
        tikimt_values = request.form.getlist('tikimt')
        hidar_values = request.form.getlist('hidar')
        tahsas_values = request.form.getlist('tahsas')
        tir_values = request.form.getlist('tir')
        yekatit_values = request.form.getlist('yekatit')
        megabit_values = request.form.getlist('megabit')
        miazia_values = request.form.getlist('miazia')
        ginbot_values = request.form.getlist('ginbot')
        sene_values = request.form.getlist('sene')
        
        for i, name in enumerate(activity_names):
            if name.strip():
                activity = PeaceClubActivity(
                    club_plan_id=plan.id,
                    activity_number=i + 1,
                    activity_name=name.strip(),
                    hamle=str(i) in hamle_values,
                    nehase=str(i) in nehase_values,
                    meskerem=str(i) in meskerem_values,
                    tikimt=str(i) in tikimt_values,
                    hidar=str(i) in hidar_values,
                    tahsas=str(i) in tahsas_values,
                    tir=str(i) in tir_values,
                    yekatit=str(i) in yekatit_values,
                    megabit=str(i) in megabit_values,
                    miazia=str(i) in miazia_values,
                    ginbot=str(i) in ginbot_values,
                    sene=str(i) in sene_values
                )
                db.session.add(activity)
        
        student_names = request.form.getlist('student_name[]')
        student_grades = request.form.getlist('student_grade[]')
        student_data = []
        for i in range(len(student_names)):
            if student_names[i].strip():
                student_data.append({
                    'name': student_names[i].strip(),
                    'grade': student_grades[i] if i < len(student_grades) else ''
                })
        plan.student_members = json.dumps(student_data)
        
        teacher_names = request.form.getlist('teacher_name[]')
        teacher_grades = request.form.getlist('teacher_grade[]')
        teacher_data = []
        for i in range(len(teacher_names)):
            if teacher_names[i].strip():
                teacher_data.append({
                    'name': teacher_names[i].strip(),
                    'grade': teacher_grades[i] if i < len(teacher_grades) else ''
                })
        plan.teacher_members = json.dumps(teacher_data)
        
        db.session.commit()
        flash('Peace Club plan updated successfully!', 'success')
        return redirect(url_for('peaceclub_view', plan_id=plan.id))
    
    return render_template('peaceclub_edit.html', 
                         plan=plan, 
                         activities=activities,
                         student_members=student_members,
                         teacher_members=teacher_members)

@app.route('/peaceclub/delete/<int:plan_id>')
def peaceclub_delete(plan_id):
    plan = PeaceClubPlan.query.get_or_404(plan_id)
    activities = PeaceClubActivity.query.filter_by(club_plan_id=plan_id).all()
    for activity in activities:
        db.session.delete(activity)
    db.session.delete(plan)
    db.session.commit()
    flash('Peace Club plan deleted successfully!', 'success')
    return redirect(url_for('peaceclub_home'))

# ================================================================
# CREATE TABLES ON APPLICATION STARTUP
# ================================================================
with app.app_context():
    try:
        db.create_all()
        print("✅ Database tables created/verified successfully.")
        print(f"📊 Using database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")

# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
