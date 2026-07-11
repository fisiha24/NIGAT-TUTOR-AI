import os
import json
import re
import requests
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from werkzeug.utils import secure_filename
from datetime import datetime
from io import BytesIO
from docx import Document

# ================================================================
# USE GROQ ONLY (GEMINI DISABLED)
# ================================================================
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

app = Flask(__name__)

# ================================================================
# DATABASE CONFIGURATION - POSTGRESQL FOR RENDER
# ================================================================
database_url = os.environ.get('DATABASE_URL')

if database_url:
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    print("✅ Using PostgreSQL database from Render")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nigat.db'
    print("⚠️ Using SQLite (local development mode)")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'mysecretkey')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# --- Upload folder initialization ---
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ================================================================
# GROQ API KEY
# ================================================================
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', "")

# Initialize Groq client
groq_client = None

if GROQ_AVAILABLE and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialized successfully")
    except Exception as e:
        print(f"⚠️ Groq init failed: {e}")
else:
    print("⚠️ Groq not configured - check API key")

db = SQLAlchemy(app)

# --- In-memory storage for uploaded content ---
uploaded_texts = {
    'pdf': [],
    'images': []
}

# ================================================================
# MODELS
# ================================================================
class Course(db.Model):
    __tablename__ = 'course'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    color = db.Column(db.String(20), default='#764ba2')
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

# --- Admin Views ---
admin = Admin(app, name='Nigat Admin')
admin.add_view(ModelView(Course, db))
admin.add_view(ModelView(AnnualPlan, db))
admin.add_view(ModelView(LaboratoryPlan, db))
admin.add_view(ModelView(DailyPlan, db))

# ================================================================
# DATABASE TABLE CREATION - CRITICAL FIX
# ================================================================
def init_db():
    """Initialize database tables and add sample data"""
    try:
        with app.app_context():
            db.create_all()
            print("✅ Database tables created/verified successfully.")
            print(f"📊 Using database: {app.config['SQLALCHEMY_DATABASE_URI']}")
            
            # Add sample courses if none exist
            if Course.query.count() == 0:
                print("📝 Adding sample course data...")
                sample_courses = [
                    Course(name="Mathematics", description="Algebra, Geometry, Calculus and more", color="#667eea"),
                    Course(name="Physics", description="Mechanics, Thermodynamics, Optics and more", color="#48bb78"),
                    Course(name="Chemistry", description="Elements, Compounds, Reactions and more", color="#ed8936"),
                    Course(name="Biology", description="Living organisms, Ecosystems, Genetics and more", color="#38b2ac"),
                    Course(name="English", description="Grammar, Literature, Writing and more", color="#9f7aea"),
                    Course(name="History", description="World History, Ethiopian History and more", color="#fc8181")
                ]
                for course in sample_courses:
                    db.session.add(course)
                db.session.commit()
                print("✅ Sample courses added successfully!")
                
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        print(f"⚠️ Database URL: {app.config['SQLALCHEMY_DATABASE_URI']}")
        return False
    return True

# Initialize database on startup
with app.app_context():
    init_db()

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def detect_language(text):
    """Detect if text is Amharic or English based on Unicode range"""
    if not text:
        return 'english'
    amharic_pattern = re.compile(r'[\u1200-\u137F]')
    if amharic_pattern.search(text):
        return 'amharic'
    return 'english'

def summarize_for_context(text, max_chars=3000):
    if len(text) <= max_chars:
        return text
    first_part = text[:int(max_chars * 0.7)]
    last_part = text[-int(max_chars * 0.3):]
    return f"{first_part}\n\n[...truncated...]\n\n{last_part}"

def extract_pdf_text(filepath):
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text if text else "No text found in PDF."
    except Exception as e:
        return f"PDF extraction error: {str(e)}"

def remove_duplicate_sentences(text):
    """Remove duplicate sentences from AI response"""
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
        
        if len(sentence) < 5 or sentence in ['', ' ', '...']:
            continue
            
        seen.add(normalized)
        unique_sentences.append(sentence)
    
    result = ' '.join(unique_sentences)
    
    if len(result) < 20 and len(text) > 50:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        unique = []
        seen = set()
        for s in sentences:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                unique.append(s)
                if len(unique) >= 3:
                    break
        if unique:
            result = ' '.join(unique)
    
    return result

# ================================================================
# AI RESPONSE FUNCTION (GROQ ONLY)
# ================================================================
def get_ai_response(system_prompt, user_query):
    """Get response from Groq API only"""
    
    if groq_client is None:
        return "⚠️ Groq API is not available. Please check your API key."
    
    try:
        print("🤖 Using Groq API (llama-3.3-70b-versatile)...")
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.05,
            max_tokens=2048,
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
# ROUTES
# ================================================================
@app.route('/')
def home():
    try:
        courses = Course.query.all()
        return render_template('index.html', courses=courses)
    except Exception as e:
        print(f"❌ Home route error: {e}")
        # Try to recreate tables if they don't exist
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
# UPLOAD ROUTES
# ================================================================
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file uploaded.', 'error')
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('home'))
    
    if file and file.filename.lower().endswith('.pdf'):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            text = extract_pdf_text(filepath)
            
            if file_size > 10:
                max_chars = 5000
            elif file_size > 5:
                max_chars = 4000
            else:
                max_chars = 3000
            
            truncated_text = summarize_for_context(text, max_chars=max_chars)
            uploaded_texts['pdf'] = []
            uploaded_texts['pdf'].append(truncated_text)
            
            flash(f'PDF uploaded successfully! ({file_size:.1f}MB)', 'success')
            return redirect(url_for('home'))
        except Exception as e:
            flash(f'Error processing PDF: {str(e)}', 'error')
            return redirect(url_for('home'))
    else:
        flash('Only PDF files are allowed.', 'error')
        return redirect(url_for('home'))

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
        uploaded_texts['images'].append(f"Image uploaded: {filename}")
        return jsonify({'message': 'Image uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

@app.route('/clear_context', methods=['POST'])
def clear_context():
    uploaded_texts['pdf'] = []
    uploaded_texts['images'] = []
    return jsonify({'message': 'Context cleared'}), 200

# ================================================================
# AI CHAT ROUTE
# ================================================================
@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    user_query = request.json.get('query', '').strip()
    
    if not user_query:
        return jsonify({"answer": "Please ask a question."})
    
    # Detect language of the query
    query_lang = detect_language(user_query)
    print(f"🔍 Detected language: {query_lang}")
    
    # Build context
    context = ""
    if uploaded_texts['pdf']:
        pdf_text = "\n".join(uploaded_texts['pdf'][-2:])
        if len(pdf_text) > 2500:
            pdf_text = pdf_text[:2500] + "... [truncated]"
        context += "PDF Content:\n" + pdf_text + "\n"
    
    if uploaded_texts['images']:
        img_text = "\n".join(uploaded_texts['images'][-2:])
        context += "Image Uploaded:\n" + img_text + "\n"
    
    # LANGUAGE INSTRUCTION
    if query_lang == 'amharic':
        language_instruction = "You MUST respond in Amharic (በአማርኛ)."
    else:
        language_instruction = "You MUST respond in English."
    
    # SYSTEM PROMPT
    system_prompt = (
        "You are 'Nigat AI Tutor'. Your creator is Teacher Fisaha Melke.\n\n"
        
        "=== LANGUAGE RULE ===\n"
        f"{language_instruction}\n"
        "Do NOT switch languages. The response MUST be in the same language as the user's question.\n\n"
        
        "=== CONTEXT ===\n"
        f"{context}\n\n"
        
        "=== FIXED RESPONSES ===\n"
        "1. If asked about speaking Amharic:\n"
        "   - English: 'Yes, I can speak Amharic fluently. I can help you with any question.'\n"
        "   - Amharic: 'አዎ፣ እኔ አማርኛን በደንብ እናገራለሁ። በማንኛውም ጥያቄ ልረዳህ እችላለሁ።'\n\n"
        
        "2. If asked 'Who created you?':\n"
        "   - Amharic: 'እኔን የሰራኝ መምህር ፍስሃ መልኬ ይባላል። እሱ የሁለት ዓመት የመማር እና ማስተማር ልምድ አለው። በቪዲዮ ኢዲቲንግ ዘርፍም ሰርቲፊኬት አለው። ለተማሪዎች በቤት ለቤት ትምህርት እና የጥናት ድጋፍ ይሰጣል። ማንኛውም መረጃ ወይም ግንኙነት ለማግኘት በሚከተሉት ስልክ ቁጥሮች መደወል ይቻላል፦ 0919 704 062 / 0978 127 213 አዲስ አበባ ከተማ ውስጥ ይገኛል።'\n"
        "   - English: 'I was created by Teacher Fisaha Melke. He has two years of experience in teaching and learning activities. He also holds a certificate in video editing. He provides home-to-home tutoring and academic support for students. For more information or contact, you can call: 0919 704 062 / 0978 127 213. He is based in Addis Ababa.'\n\n"
        
        "=== CRITICAL: TABLE FORMATTING RULES ===\n"
        "You MUST format ALL tables with proper line breaks and Markdown syntax.\n"
        "Each row of a table MUST be on a NEW LINE.\n"
        "Example of CORRECT table format:\n"
        "| Column 1 | Column 2 | Column 3 |\n"
        "|----------|----------|----------|\n"
        "| Data 1   | Data 2   | Data 3   |\n"
        "| Data 4   | Data 5   | Data 6   |\n\n"
        "REMEMBER: Every table row must be on its own separate line.\n\n"
        
        "=== DAILY LESSON PLAN TEMPLATE ===\n"
        "3. When asked to generate a DAILY LESSON PLAN, use this EXACT TEMPLATE with placeholders and TABLES:\n\n"
        "# SCHOOL INFORMATION\n"
        "**School Name:** [SCHOOL_NAME]\n"
        "**Teacher Name:** [TEACHER_NAME]\n"
        "**Grade and Section:** [GRADE_AND_SECTION]\n"
        "**Subject:** [SUBJECT]\n"
        "**Date:** [DATE]\n"
        "**Unit:** [UNIT_NUMBER - UNIT_TITLE]\n"
        "**Lesson Topic:** [LESSON_TOPIC]\n"
        "**Page:** [PAGE]\n\n"
        "# LESSON OVERVIEW\n"
        "**Rationale of the topic:** [RATIONALE]\n"
        "**Pre-requisite Knowledge:** [PREREQUISITES]\n"
        "**Competencies (Learning Objectives):**\n"
        "- [COMPETENCY_1]\n"
        "- [COMPETENCY_2]\n"
        "- [COMPETENCY_3]\n\n"
        "# LESSON STAGES (TABLE)\n"
        "| Stage | Time | Learning Contents | Page | Teacher Activities | Student Activities | Teaching Methodology | Learning Assessment | Teaching Aids | Remark |\n"
        "|-------|------|-------------------|------|-------------------|---------------------|----------------------|---------------------|---------------|--------|\n"
        "| Starter / Introduction | [TIME] | [CONTENT] | [PAGE] | [TEACHER_ACTIVITIES] | [STUDENT_ACTIVITIES] | [METHODOLOGY] | [ASSESSMENT] | [AIDS] | [REMARK] |\n"
        "| Main Activities | [TIME] | [CONTENT] | [PAGE] | [TEACHER_ACTIVITIES] | [STUDENT_ACTIVITIES] | [METHODOLOGY] | [ASSESSMENT] | [AIDS] | [REMARK] |\n"
        "| Concluding Activities | [TIME] | [CONTENT] | [PAGE] | [TEACHER_ACTIVITIES] | [STUDENT_ACTIVITIES] | [METHODOLOGY] | [ASSESSMENT] | [AIDS] | [REMARK] |\n\n"
        "# SUPPORT FOR LEARNERS WITH SPECIAL NEEDS (TABLE)\n"
        "| Category | Support Strategies |\n"
        "|----------|-------------------|\n"
        "| Slow-learners | [SLOW_LEARNERS_STRATEGIES] |\n"
        "| Medium-learners | [MEDIUM_LEARNERS_STRATEGIES] |\n"
        "| Fast-learners | [FAST_LEARNERS_STRATEGIES] |\n\n"
        "# APPROVALS (TABLE)\n"
        "| Role | Name | Signature | Date |\n"
        "|------|------|-----------|------|\n"
        "| Teacher | [TEACHER_NAME] | [TEACHER_SIGNATURE] | [DATE] |\n"
        "| Department Head | [DEPT_HEAD_NAME] | [DEPT_HEAD_SIGNATURE] | [DATE] |\n"
        "| Vice Principal | [VP_NAME] | [VP_SIGNATURE] | [DATE] |\n\n"
        "# POST-LESSON TEACHER'S SELF-ASSESSMENT\n"
        "[SELF_ASSESSMENT]\n\n"
        
        "=== ANNUAL LESSON PLAN TEMPLATE ===\n"
        "4. When asked to generate an ANNUAL LESSON PLAN, use this EXACT TEMPLATE with TABLES:\n\n"
        "# ANNUAL LESSON PLAN\n"
        "**School Name:** [SCHOOL_NAME]\n"
        "**Teacher Name:** [TEACHER_NAME]\n"
        "**Subject:** [SUBJECT]\n"
        "**Grade and Section:** [GRADE_AND_SECTION]\n"
        "**Academic Year:** [YEAR]\n"
        "**Total Working Days:** [TOTAL_DAYS]\n"
        "**1st Semester Days:** [SEM1_DAYS]\n"
        "**2nd Semester Days:** [SEM2_DAYS]\n\n"
        "**Unit [UNIT_NUMBER]: [UNIT_TITLE]**\n"
        "**General Objectives:** [UNIT_OBJECTIVES]\n\n"
        "| Month | Week | Period | Date Range | Page | Topics | Objectives | Methodology | Teaching Aids | Evaluation |\n"
        "|-------|------|--------|------------|------|--------|------------|-------------|---------------|------------|\n"
        "| [MONTH_1] | [WEEK_1] | [PERIOD_1] | [DATE_RANGE_1] | [PAGE_1] | [TOPICS_1] | [OBJECTIVES_1] | [METHOD_1] | [AIDS_1] | [EVALUATION_1] |\n"
        "| [MONTH_2] | [WEEK_2] | [PERIOD_2] | [DATE_RANGE_2] | [PAGE_2] | [TOPICS_2] | [OBJECTIVES_2] | [METHOD_2] | [AIDS_2] | [EVALUATION_2] |\n\n"
        "**Prepared By:** [PREPARER_NAME]\n"
        "**Department Head:** [DEPT_HEAD_NAME]\n"
        "**Director:** [DIRECTOR_NAME]\n"
        "**Signatures & Dates:** ...\n\n"
        
        "=== LABORATORY PLAN TEMPLATE ===\n"
        "5. When asked to generate a LABORATORY PLAN, use this EXACT TEMPLATE with TABLES:\n\n"
        "# LABORATORY ANNUAL PLAN\n"
        "**School Name:** [SCHOOL_NAME]\n"
        "**Teacher Name:** [TEACHER_NAME]\n"
        "**Subject:** [SUBJECT]\n"
        "**Grade:** [GRADE]\n"
        "**Academic Year:** [YEAR]\n\n"
        "| Experiment No. | Title | Apparatus | Chemicals | Unit | Page | Month | Date |\n"
        "|---------------|-------|-----------|-----------|------|------|-------|------|\n"
        "| [EXP_1] | [TITLE_1] | [APPARATUS_1] | [CHEMICALS_1] | [UNIT_1] | [PAGE_1] | [MONTH_1] | [DATE_1] |\n"
        "| [EXP_2] | [TITLE_2] | [APPARATUS_2] | [CHEMICALS_2] | [UNIT_2] | [PAGE_2] | [MONTH_2] | [DATE_2] |\n\n"
        "**Prepared By:** [PREPARER_NAME]\n"
        "**Approved By:** [APPROVER_NAME]\n"
        "**Dates:** ...\n\n"
        
        "=== ACCURACY RULE ===\n"
        "Provide ONLY accurate information. If you don't know, say: 'I don't have accurate information about that.' in the user's language.\n\n"
        
        "=== AMHARIC SPELLING ===\n"
        "Correct spellings: 'ጎንደር' (not ንንደር/ጀንደር), 'ኢትዮጵያ' (not እትዮጵያ).\n\n"
        
        "=== FINAL REMINDER ===\n"
        "1. TABLES MUST HAVE PROPER LINE BREAKS. Each row on a new line.\n"
        "2. For Amharic responses, use correct Amharic spelling and script.\n"
        "3. If the user asks in English, respond in English with all tables in English. If in Amharic, respond in Amharic with all tables in Amharic.\n"
        "4. NEVER repeat sentences. Write each sentence only ONCE.\n"
        "5. Write 3-5 sentences for general questions.\n"
        "6. When the user asks for a lesson plan, generate the complete template with ALL sections above.\n"
        "7. Do NOT change the format or remove any sections.\n"
        "8. The user should fill in the placeholders [LIKE_THIS] with their own information."
    )
    
    answer = get_ai_response(system_prompt, user_query)
    
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
                doc.add_paragraph(line)
    
    if in_table and table_rows:
        num_cols = max(len(table_headers), max([len(row) for row in table_rows]) if table_rows else 0)
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
    
    return send_file(
        file_stream,
        as_attachment=True,
        download_name=f"{filename}.docx",
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )

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
# MAIN
# ================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
