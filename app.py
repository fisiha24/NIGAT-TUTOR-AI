import os
import json
import re
import uuid
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
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ================================================================
# GROQ API - MULTIPLE KEYS (ONLY GROQ)
# ================================================================
GROQ_API_KEYS = []
for i in range(1, 4):  # GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3
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
uploaded_texts = {'pdf': [], 'images': []}

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

def summarize_for_context(text, max_chars=4000):
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

def get_ai_response(system_prompt, user_query):
    """Get response from Groq API using multiple keys (round-robin)"""
    
    try:
        from groq import Groq
        key = get_next_groq_key()
        if key is None:
            return "⚠️ No Groq API keys available. Please add at least one API key."
        
        client = Groq(api_key=key)
        print("🤖 Using Groq API (llama-3.3-70b-versatile)...")
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.05,
            max_tokens=512,
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
# UPLOAD ROUTES
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
            
            if file_size > 30:
                max_chars = 3000
            elif file_size > 15:
                max_chars = 2500
            else:
                max_chars = 4000
            
            truncated_text = summarize_for_context(text, max_chars=max_chars)
            
            if 'session_id' not in session:
                session['session_id'] = str(uuid.uuid4())
            
            session['pdf_context'] = truncated_text
            session['pdf_filename'] = filename
            session['pdf_size'] = file_size
            
            uploaded_texts['pdf'] = []
            uploaded_texts['pdf'].append(truncated_text)
            
            return jsonify({
                'success': True, 
                'message': f'PDF uploaded successfully! ({file_size:.1f}MB)',
                'session_id': session.get('session_id')
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
        uploaded_texts['images'].append(f"Image uploaded: {filename}")
        return jsonify({'message': 'Image uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

@app.route('/clear_context', methods=['POST'])
def clear_context():
    uploaded_texts['pdf'] = []
    uploaded_texts['images'] = []
    session.pop('pdf_context', None)
    session.pop('pdf_filename', None)
    session.pop('pdf_size', None)
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
    
    context = ""
    
    pdf_text = session.get('pdf_context', '')
    if pdf_text:
        if len(pdf_text) > 3500:
            pdf_text = pdf_text[:3500] + "... [truncated]"
        context += "PDF Content:\n" + pdf_text + "\n"
        print(f"📄 Using PDF from session: {session.get('pdf_filename', 'unknown')}")
    
    elif uploaded_texts['pdf']:
        pdf_text = "\n".join(uploaded_texts['pdf'][-2:])
        if len(pdf_text) > 3500:
            pdf_text = pdf_text[:3500] + "... [truncated]"
        context += "PDF Content:\n" + pdf_text + "\n"
        print("📄 Using PDF from uploaded_texts")
    
    if uploaded_texts['images']:
        img_text = "\n".join(uploaded_texts['images'][-2:])
        context += "Image Uploaded:\n" + img_text + "\n"
    
    if query_lang == 'amharic':
        language_instruction = "You MUST respond in Amharic (በአማርኛ)."
    else:
        language_instruction = "You MUST respond in English."
    
    system_prompt = (
        "You are 'Nigat AI Tutor'. Your creator is Teacher Fisaha Melke.\n\n"
        
        "=== LANGUAGE RULE ===\n"
        f"{language_instruction}\n"
        "Do NOT switch languages. The response MUST be in the same language as the user's question.\n\n"
        
        "=== CONTEXT ===\n"
        f"{context}\n\n"
        
        "=== ABOUT THE CREATOR ===\n"
        "When asked about the creator, respond with the following information:\n"
        "English:\n"
        "My name is Fisiha Melke. I graduated from Ambo University with a Bachelor's degree in Biology in 2024 (2016 E.C.). I have more than two years of teaching experience in private schools, where I have been dedicated to helping students learn through clear explanations, practical examples, and student-centered teaching methods.\n\n"
        "In addition to teaching, I hold a Certificate in Video Editing, which has strengthened my ability to create engaging digital educational content. I believe that combining education with technology creates more effective and enjoyable learning experiences.\n\n"
        "I am currently developing Nigat Tutor AI, an educational platform designed to support Ethiopian teachers and students. The platform aims to provide AI-powered learning assistance, lesson plan generation, exam and worksheet creation, PDF-based learning support, educational content generation, and intelligent tutoring in both Amharic and English.\n\n"
        "My vision is to make quality education more accessible by using modern technology to reduce teachers' workload, improve students' learning outcomes, and promote innovative digital learning across Ethiopia.\n\n"
        "I am a hardworking, self-motivated, and lifelong learner who enjoys solving problems and continuously improving my skills. I am committed to contributing to the future of education by developing practical and innovative educational technologies that benefit teachers, students, and schools.\n\n"
        "Amharic (አማርኛ):\n"
        "ስሜ ፍስሃ መልኬ ነው። ከአምቦ ዩኒቨርሲቲ በባዮሎጂ (Biology) ትምህርት ዘርፍ በ2016 ዓ.ም. ተመርቄያለሁ። ከሁለት ዓመት በላይ በግል ትምህርት ቤቶች ውስጥ የማስተማር ልምድ አለኝ፣ በተለይም ተማሪዎችን በቀላል፣ በግልጽ እና በተግባራዊ መንገድ ማስተማር ላይ ትኩረት አደርጋለሁ።\n\n"
        "ከማስተማር ሙያዬ በተጨማሪ በቪዲዮ ኤዲቲንግ (Video Editing) ዘርፍ የሙያ ሰርቲፊኬት አለኝ። ቴክኖሎጂን ከትምህርት ጋር በማጣመር የተማሪዎችን የመማር ልምድ ለማሻሻል እጥራለሁ።\n\n"
        "በአሁኑ ጊዜ የኢትዮጵያ መምህራንና ተማሪዎችን ለማገዝ Nigat Tutor AI የተባለ ዘመናዊ የትምህርት መድረክ እያዘጋጀሁ ነው። ይህ መድረክ የትምህርት እቅድ ማዘጋጀት፣ ጥያቄዎችን ማመንጨት፣ ፈተና ማዘጋጀት፣ ከPDF መጽሐፍት መረጃ ማውጣት፣ እንዲሁም በአማርኛና በእንግሊዝኛ የAI እገዛ መስጠት የሚችል ነው።\n\n"
        "የእኔ ዓላማ ቴክኖሎጂን በመጠቀም የትምህርት ጥራትን ማሻሻል፣ የመምህራንን የስራ ጫና ማቃለል፣ እና ተማሪዎች ዘመናዊና ተደራሽ የመማሪያ መሳሪያዎችን እንዲጠቀሙ ማድረግ ነው።\n\n"
        "ታታሪ፣ ለመማር ፈቃደኛ፣ ችግር ፈቺ እና ውጤት ተኮር ሰው ነኝ። በቀጣይም በትምህርትና በቴክኖሎጂ ዘርፍ የሚጠቅሙ ፈጠራዎችን ለማበርከት እሰራለሁ።\n\n"
        
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
        
        "=== PEACE CLUB PLAN TEMPLATE ===\n"
        "6. When asked to generate a PEACE CLUB PLAN, use this EXACT TEMPLATE with TABLES:\n\n"
        "# PEACE CLUB ANNUAL PLAN\n"
        "**School Name:** [SCHOOL_NAME]\n"
        "**District:** [DISTRICT]\n"
        "**Woreda:** [WOREDA]\n"
        "**School Level:** [SCHOOL_LEVEL]\n"
        "**Club Name:** [CLUB_NAME]\n"
        "**Teacher Name:** [TEACHER_NAME]\n"
        "**Secretary Name:** [SECRETARY_NAME]\n"
        "**Year:** [YEAR]\n"
        "**Month:** [MONTH]\n\n"
        "**Vision:** [VISION]\n"
        "**Mission:** [MISSION]\n"
        "**Opportunities & Strengths:** [OPPORTUNITIES]\n"
        "**Challenges & Weaknesses:** [CHALLENGES]\n"
        "**Solutions:** [SOLUTIONS]\n\n"
        "| # | Activity | Hamle | Nehase | Meskerem | Tikimt | Hidar | Tahsas | Tir | Yekatit | Megabit | Miazia | Ginbot | Sene |\n"
        "|---|----------|-------|--------|----------|--------|-------|--------|-----|---------|---------|--------|--------|------|\n"
        "| 1 | [ACTIVITY_1] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] |\n"
        "| 2 | [ACTIVITY_2] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] |\n\n"
        "**Student Members:** [LIST_OF_STUDENTS]\n"
        "**Teacher Members:** [LIST_OF_TEACHERS]\n\n"
        
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
