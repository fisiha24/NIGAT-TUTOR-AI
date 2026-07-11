import os
import json
import re
import uuid
import hashlib
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
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB - አይቀየርም!
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ================================================================
# GROQ API
# ================================================================
GROQ_API_KEYS = []
for i in range(1, 4):
    key = os.environ.get(f'GROQ_API_KEY_{i}', '')
    if key:
        GROQ_API_KEYS.append(key)

main_groq_key = os.environ.get('GROQ_API_KEY', '')
if main_groq_key and main_groq_key not in GROQ_API_KEYS:
    GROQ_API_KEYS.append(main_groq_key)

current_key_index = 0

def get_next_groq_key():
    global current_key_index
    if not GROQ_API_KEYS:
        return None
    key = GROQ_API_KEYS[current_key_index % len(GROQ_API_KEYS)]
    current_key_index += 1
    return key

print(f"✅ Loaded {len(GROQ_API_KEYS)} Groq API keys")

db = SQLAlchemy(app)

# ================================================================
# SMART CONTEXT MANAGEMENT
# ================================================================

class DocumentChunk:
    """የPDF ክፍልፋዮችን ለማስተዳደር"""
    def __init__(self, text, page_num, chunk_id):
        self.text = text
        self.page_num = page_num
        self.chunk_id = chunk_id
        self.keywords = self._extract_keywords(text)
    
    def _extract_keywords(self, text):
        """ቁልፍ ቃላትን ከጽሑፉ ያውጣል"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return set(words[:50])

class DocumentStore:
    """የPDF ጽሑፎችን በማህደረ ትውስታ ውስጥ ያስቀምጣል"""
    def __init__(self):
        self.documents = {}  # session_id -> full_text
        self.chunks = {}     # session_id -> list of DocumentChunk
        self.chunk_size = 1000  # ፊደላት በአንድ ክፍል
    
    def store_document(self, session_id, text, filename):
        """ሙሉ ጽሑፉን ያስቀምጣል እና በክፍል ይከፋፍላል"""
        self.documents[session_id] = {
            'text': text,
            'filename': filename,
            'length': len(text),
            'pages': text.count('\n\n') + 1
        }
        
        # ጽሑፉን በክፍል መከፋፈል
        chunks = []
        words = text.split()
        for i in range(0, len(words), self.chunk_size):
            chunk_text = ' '.join(words[i:i+self.chunk_size])
            chunk_id = f"chunk_{i//self.chunk_size + 1}"
            page_num = (i // (self.chunk_size * 5)) + 1  # ግምታዊ ገጽ
            chunks.append(DocumentChunk(chunk_text, page_num, chunk_id))
        
        self.chunks[session_id] = chunks
        print(f"📚 Stored {len(chunks)} chunks for session {session_id}")
        return len(chunks)
    
    def get_relevant_chunks(self, session_id, query, top_k=3):
        """ከጥያቄው ጋር የሚዛመዱ ክፍሎችን ያገኛል"""
        if session_id not in self.chunks:
            return []
        
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        if not query_words:
            # ምንም ቁልፍ ቃል ከሌለ የመጀመሪያዎቹን ክፍሎች ይመልሳል
            return self.chunks[session_id][:top_k]
        
        # የውጤት ውጤት አስላ
        scored_chunks = []
        for chunk in self.chunks[session_id]:
            common = len(query_words & chunk.keywords)
            score = common / max(len(query_words), 1)
            if score > 0:
                scored_chunks.append((score, chunk))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]
    
    def get_full_text(self, session_id):
        """ሙሉ ጽሑፉን ይመልሳል"""
        if session_id in self.documents:
            return self.documents[session_id]['text']
        return None
    
    def clear(self, session_id):
        """የአንድን ክፍለ ጊዜ መረጃ ያጸዳል"""
        if session_id in self.documents:
            del self.documents[session_id]
        if session_id in self.chunks:
            del self.chunks[session_id]

# Global document store
doc_store = DocumentStore()

# ================================================================
# SMART PROMPT SYSTEM
# ================================================================

class PromptManager:
    """የተለያዩ ጥያቄዎችን ለማስተዳደር"""
    
    BASE_SYSTEM = "You are 'Nigat AI Tutor'. Created by Teacher Fisaha Melke."
    
    PROMPTS = {
        'daily_lesson': """
=== DAILY LESSON PLAN TEMPLATE ===
Generate a daily lesson plan using this format:

# SCHOOL INFORMATION
**School Name:** [SCHOOL_NAME]
**Teacher Name:** [TEACHER_NAME]
**Grade and Section:** [GRADE_AND_SECTION]
**Subject:** [SUBJECT]
**Date:** [DATE]
**Unit:** [UNIT_NUMBER - UNIT_TITLE]
**Lesson Topic:** [LESSON_TOPIC]
**Page:** [PAGE]

# LESSON OVERVIEW
**Rationale:** [RATIONALE]
**Pre-requisite Knowledge:** [PREREQUISITES]
**Competencies:** [LIST]

# LESSON STAGES (TABLE)
| Stage | Time | Teacher Activities | Student Activities | Methodology | Assessment |

# SUPPORT FOR LEARNERS
| Category | Support Strategies |

# APPROVALS (TABLE)
| Role | Name | Signature | Date |

# TEACHER'S SELF-ASSESSMENT
[SELF_ASSESSMENT]
""",
        
        'annual_plan': """
=== ANNUAL LESSON PLAN TEMPLATE ===
Generate an annual lesson plan using this format:

# ANNUAL LESSON PLAN
**School Name:** [SCHOOL_NAME]
**Teacher Name:** [TEACHER_NAME]
**Subject:** [SUBJECT]
**Grade:** [GRADE]
**Academic Year:** [YEAR]

| Month | Week | Topics | Objectives | Methodology | Evaluation |
""",
        
        'exam': """
=== EXAM GENERATOR ===
Generate exam questions based on the provided content.
Include multiple choice, true/false, and short answer questions.
"""
    }
    
    @staticmethod
    def get_prompt(prompt_type):
        """የሚፈለገውን ፕሮምፕት ይመልሳል"""
        return PromptManager.PROMPTS.get(prompt_type, "")
    
    @staticmethod
    def detect_prompt_type(query):
        """ከጥያቄው ውስጥ የፕሮምፕት አይነት ይለያል"""
        query_lower = query.lower()
        if any(word in query_lower for word in ['daily lesson', 'ዕለታዊ', 'lesson plan']):
            return 'daily_lesson'
        elif any(word in query_lower for word in ['annual', 'ዓመታዊ', 'yearly']):
            return 'annual_plan'
        elif any(word in query_lower for word in ['exam', 'test', 'quiz', 'ፈተና', 'ምዘና']):
            return 'exam'
        return 'general'

# ================================================================
# TOKEN COUNTER (ለGroq ገደብ ለመከታተል)
# ================================================================
def estimate_tokens(text):
    """ግምታዊ የቶከን ብዛት ያሰላል"""
    return len(text) // 4  # ሞቃታማ ግምት

def truncate_to_limit(text, max_tokens=5000):
    """ጽሑፉን ወደተፈለገው ገደብ ያሳጥራል"""
    if estimate_tokens(text) <= max_tokens:
        return text
    chars = max_tokens * 4
    return text[:chars] + "\n\n[...truncated...]"

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

def extract_pdf_text(filepath):
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text if text else "No text found in PDF."
    except Exception as e:
        return f"PDF extraction error: {str(e)}"

def remove_duplicate_sentences(text):
    if not text:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        normalized = sentence.lower()
        if normalized in seen:
            continue
        if len(sentence) < 5:
            continue
        seen.add(normalized)
        unique_sentences.append(sentence)
    return ' '.join(unique_sentences)

# ================================================================
# AI RESPONSE FUNCTION (የተሻሻለ)
# ================================================================
def get_ai_response(system_prompt, user_query, context):
    """Smart context management ያለው AI ጥሪ"""
    
    try:
        from groq import Groq
        key = get_next_groq_key()
        if key is None:
            return "⚠️ No Groq API keys available."
        
        # 1. የፕሮምፕት አይነት መለየት
        prompt_type = PromptManager.detect_prompt_type(user_query)
        prompt_template = PromptManager.get_prompt(prompt_type)
        
        # 2. የቶከን ገደብ ማስላት
        full_prompt = f"{system_prompt}\n\n{prompt_template}\n\nContext:\n{context}\n\nUser: {user_query}"
        estimated_tokens = estimate_tokens(full_prompt)
        
        print(f"📊 Estimated tokens: {estimated_tokens}")
        
        # 3. ከገደብ በላይ ከሆነ አሳጥር
        if estimated_tokens > 5000:
            print("⚠️ Reducing context to fit token limit...")
            context = truncate_to_limit(context, 3000)
            full_prompt = f"{system_prompt}\n\n{prompt_template}\n\nContext:\n{context}\n\nUser: {user_query}"
        
        client = Groq(api_key=key)
        print(f"🤖 Using Groq API ({prompt_type} prompt)...")
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.05,
            max_tokens=1024,
            top_p=0.85,
        )
        
        if chat_completion and chat_completion.choices:
            print("✅ Groq response received")
            return chat_completion.choices[0].message.content
            
    except Exception as e:
        print(f"⚠️ Groq error: {e}")
        return f"AI Error: {str(e)}"
    
    return "⚠️ No response from AI service."

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
# UPLOAD ROUTES (አይቀየርም - 500MB ነው)
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
            text = extract_pdf_text(filepath)
            
            # ሙሉ ጽሑፉን በDocumentStore ውስጥ ማስቀመጥ
            if 'session_id' not in session:
                session['session_id'] = str(uuid.uuid4())
            
            session_id = session['session_id']
            
            # ሙሉ ጽሑፉን በክፍል ማስቀመጥ
            num_chunks = doc_store.store_document(session_id, text, filename)
            
            # ለጊዜው አጭር ማስታወሻ በsession ውስጥ ማስቀመጥ
            session['pdf_filename'] = filename
            session['pdf_size'] = file_size
            session['pdf_chunks'] = num_chunks
            
            return jsonify({
                'success': True, 
                'message': f'PDF uploaded successfully! ({file_size:.1f}MB, {num_chunks} chunks)',
                'session_id': session_id,
                'chunks': num_chunks,
                'pages': text.count('\n\n') + 1
            }), 200
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error processing PDF: {str(e)}'}), 500
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
        doc_store.clear(session_id)
    session.pop('pdf_context', None)
    session.pop('pdf_filename', None)
    session.pop('pdf_size', None)
    session.pop('pdf_chunks', None)
    return jsonify({'message': 'Context cleared successfully'}), 200

# ================================================================
# AI CHAT ROUTE (የተሻሻለ - Smart Context)
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
    
    # 1. ከጥያቄው ጋር የሚዛመዱ ክፍሎችን ማግኘት
    relevant_chunks = []
    if session_id:
        relevant_chunks = doc_store.get_relevant_chunks(session_id, user_query, top_k=3)
        print(f"📚 Found {len(relevant_chunks)} relevant chunks")
    
    # 2. ተዛማጅ ጽሑፍ መገንባት
    context = ""
    if relevant_chunks:
        for chunk in relevant_chunks:
            context += f"\n--- {chunk.chunk_id} (Page ~{chunk.page_num}) ---\n{chunk.text}\n"
    elif session_id:
        # ምንም ተዛማጅ ክፍል ካልተገኘ ሙሉውን ጽሑፍ አሳጥረው ይላኩ
        full_text = doc_store.get_full_text(session_id)
        if full_text:
            context = truncate_to_limit(full_text, 3000)
            print("📄 Using full text (truncated)")
    else:
        context = "No document uploaded."
    
    # 3. የቋንቋ መመሪያ
    if query_lang == 'amharic':
        language_instruction = "You MUST respond in Amharic (በአማርኛ)."
    else:
        language_instruction = "You MUST respond in English."
    
    # 4. አጭር System Prompt
    system_prompt = (
        "You are 'Nigat AI Tutor'. Created by Teacher Fisaha Melke.\n\n"
        f"=== LANGUAGE RULE ===\n{language_instruction}\n"
        "Do NOT switch languages.\n\n"
        "=== ABOUT THE CREATOR ===\n"
        "My name is Fisiha Melke. I graduated from Ambo University with a Bachelor's degree in Biology in 2024 (2016 E.C.). I have more than two years of teaching experience in private schools. I hold a Certificate in Video Editing. I am currently developing Nigat Tutor AI, an educational platform designed to support Ethiopian teachers and students.\n\n"
        "=== ACCURACY RULE ===\n"
        "Provide ONLY accurate information. If you don't know, say: 'I don't have accurate information about that.'\n\n"
        "=== AMHARIC SPELLING ===\n"
        "Correct spellings: 'ጎንደር' (not ንንደር/ጀንደር), 'ኢትዮጵያ' (not እትዮጵያ).\n"
    )
    
    # 5. AI መልስ ማግኘት
    answer = get_ai_response(system_prompt, user_query, context)
    
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
