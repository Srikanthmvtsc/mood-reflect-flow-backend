from flask import Flask, request, jsonify, session
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import sqlite3
import os
from datetime import datetime, timedelta
import google.generativeai as genai
import logging
import hashlib
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import re
from functools import wraps
# Import DeepFace for emotion detection
from deepface import DeepFace
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
CORS(app, supports_credentials=True)

# Email configuration
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', 587))
EMAIL_USER = os.getenv('EMAIL_USER', 'your-email@gmail.com')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', 'your-app-password')
THERAPIST_EMAIL = os.getenv('THERAPIST_EMAIL', 'therapist@example.com')

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    logger.warning("GEMINI_API_KEY not found in environment variables")
    model = None

# DeepFace configuration
DEEPFACE_MODEL = 'emotion'  # Built-in emotion detection
DEEPFACE_BACKEND = 'opencv'  # Face detection backend
DEEPFACE_ENFORCE_DETECTION = False  # Don't fail if no face detected

# Emotion mapping from DeepFace output to our labels
EMOTION_MAPPING = {
    'angry': 'angry',
    'disgust': 'disgust', 
    'fear': 'fear',
    'happy': 'happy',
    'sad': 'sad',
    'surprise': 'surprise',
    'neutral': 'neutral'
}

def init_deepface():
    """Initialize DeepFace emotion detection"""
    try:
        logger.info("Initializing DeepFace emotion detection...")
        
        # Test DeepFace with a dummy image to ensure models are downloaded
        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # This will download the required models on first run
        test_result = DeepFace.analyze(
            img_path=dummy_image,
            actions=['emotion'],
            detector_backend=DEEPFACE_BACKEND,
            enforce_detection=False,
            silent=True
        )
        
        logger.info("✅ DeepFace emotion detection initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing DeepFace: {e}")
        return False

def enhance_image_for_detection(image):
    """Enhance image quality for better face detection"""
    try:
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB, enhance contrast
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            return enhanced
        else:
            return image
            
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return image

def detect_emotion_deepface(image):
    """
    Detect emotion using DeepFace
    Returns: (emotion, confidence, all_emotions_dict)
    """
    try:
        # Ensure image is in the right format
        if image is None or image.size == 0:
            logger.error("Empty image provided")
            return None, 0.0, {}
        
        # Convert to RGB if needed and enhance
        if len(image.shape) == 3:
            # Assume it's RGB from PIL
            processed_image = enhance_image_for_detection(image)
        else:
            logger.error("Invalid image format")
            return None, 0.0, {}
        
        logger.info(f"Processing image with DeepFace - Shape: {processed_image.shape}")
        
        # Analyze with DeepFace
        try:
            try:
                result = DeepFace.analyze(
                    img_path=image,
                    actions=['emotion'],
                    detector_backend=DEEPFACE_BACKEND,
                    enforce_detection=True,  # This will raise an exception if no face found
                    silent=True
                )
            except ValueError as e:
                if "Face could not be detected" in str(e):
                    return None, 0.0, {}
                raise
            
            # Handle both single result and list results
            if isinstance(result, list):
                emotion_data = result[0]['emotion'] if result else {}
            else:
                emotion_data = result['emotion']
            
            if not emotion_data:
                logger.warning("No emotion data returned from DeepFace")
                return None, 0.0, {}
            
            # Find dominant emotion
            dominant_emotion = max(emotion_data, key=emotion_data.get)
            confidence = emotion_data[dominant_emotion] / 100.0  # Convert percentage to 0-1
            
            # Map to our emotion labels
            mapped_emotion = EMOTION_MAPPING.get(dominant_emotion.lower(), dominant_emotion.lower())
            
            logger.info(f"DeepFace Results - Emotion: {mapped_emotion}, Confidence: {confidence:.3f}")
            logger.debug(f"All emotions: {emotion_data}")
            
            # Convert all emotions to our format (ensure JSON serializable)
            mapped_emotions = {}
            for emotion, score in emotion_data.items():
                mapped_key = EMOTION_MAPPING.get(emotion.lower(), emotion.lower())
                mapped_emotions[mapped_key] = float(score / 100.0)  # Convert to Python float
            
            return mapped_emotion, float(confidence), mapped_emotions
            
        except Exception as e:
            logger.error(f"DeepFace analysis failed: {e}")
            
            # Try with different detector backends as fallback
            fallback_backends = ['mtcnn', 'retinaface', 'mediapipe', 'ssd']
            
            for backend in fallback_backends:
                try:
                    logger.info(f"Trying fallback backend: {backend}")
                    result = DeepFace.analyze(
                        img_path=processed_image,
                        actions=['emotion'],
                        detector_backend=backend,
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if isinstance(result, list):
                        emotion_data = result[0]['emotion'] if result else {}
                    else:
                        emotion_data = result['emotion']
                    
                    if emotion_data:
                        dominant_emotion = max(emotion_data, key=emotion_data.get)
                        confidence = float(emotion_data[dominant_emotion] / 100.0)  # Convert to Python float
                        mapped_emotion = EMOTION_MAPPING.get(dominant_emotion.lower(), dominant_emotion.lower())
                        
                        mapped_emotions = {}
                        for emotion, score in emotion_data.items():
                            mapped_key = EMOTION_MAPPING.get(emotion.lower(), emotion.lower())
                            mapped_emotions[mapped_key] = float(score / 100.0)  # Convert to Python float
                        
                        logger.info(f"Fallback success with {backend} - Emotion: {mapped_emotion}, Confidence: {confidence:.3f}")
                        return mapped_emotion, float(confidence), mapped_emotions
                        
                except Exception as fallback_error:
                    logger.debug(f"Fallback backend {backend} also failed: {fallback_error}")
                    continue
            
            # If all backends fail, return default
            logger.warning("All DeepFace backends failed, returning neutral")
            return 'neutral', 0.5, {'neutral': 0.5}
            
    except Exception as e:
        logger.error(f"Error in DeepFace emotion detection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, 0.0, {}

def decode_base64_image(base64_string):
    """Decode base64 image with error handling"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode to bytes
        img_bytes = base64.b64decode(base64_string)
        
        # Load with PIL
        pil_img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB
        if pil_img.mode in ('RGBA', 'LA', 'P'):
            # Convert with white background
            background = Image.new('RGB', pil_img.size, (255, 255, 255))
            if pil_img.mode == 'P':
                pil_img = pil_img.convert('RGBA')
            if pil_img.mode in ('RGBA', 'LA'):
                background.paste(pil_img, mask=pil_img.split()[-1])
            else:
                background.paste(pil_img)
            pil_img = background
        elif pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(pil_img)
        
        logger.debug(f"Decoded image: {img_array.shape}, dtype: {img_array.dtype}")
        return img_array
        
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return None

# Initialize DeepFace on startup
logger.info("Initializing DeepFace emotion detection system...")
deepface_init_success = init_deepface()
if deepface_init_success:
    logger.info("✅ DeepFace emotion detection system initialized successfully")
else:
    logger.warning("⚠️ DeepFace emotion detection initialization failed")

# Database setup
def init_db():
    conn = sqlite3.connect('neuromirror.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME,
            is_verified BOOLEAN DEFAULT FALSE,
            verification_token TEXT
        )
    ''')
    
    # Create user sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_token TEXT UNIQUE NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Update mood_history to include user_id
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mood_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            all_emotions TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Update chat_history to include user_id
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT NOT NULL,
            sender TEXT NOT NULL,
            mood TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            is_flagged BOOLEAN DEFAULT FALSE,
            flag_reason TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emotion TEXT NOT NULL,
            message TEXT NOT NULL,
            tip TEXT NOT NULL,
            activity TEXT,
            sound TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create emergency alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emergency_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_email TEXT,
            username TEXT,
            alert_type TEXT NOT NULL,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_handled BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

def get_db_connection():
    conn = sqlite3.connect('neuromirror.db')
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_session_token():
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

def send_email(to_email, subject, body, is_html=False):
    """Send email using SMTP"""
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        
        if is_html:
            msg.attach(MIMEText(body, 'html'))
        else:
            msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return False

def send_welcome_email(email, username):
    """Send welcome email to new user"""
    subject = "Welcome to NeuroMirror - Your Mental Health Companion"
    body = f"""
    <html>
    <body>
        <h2>Welcome to NeuroMirror, {username}!</h2>
        <p>Thank you for joining our community dedicated to mental health and emotional well-being.</p>
        <p>Your account has been successfully created and you can now:</p>
        <ul>
            <li>Track your mood journey with AI-powered emotion detection</li>
            <li>Chat with our therapeutic AI companion</li>
            <li>Receive personalized suggestions and activities</li>
            <li>Monitor your emotional patterns over time</li>
        </ul>
        <p>Remember, your mental health matters. If you're experiencing a crisis or need immediate support, please contact a mental health professional or call your local crisis hotline.</p>
        <p>Best regards,<br>The NeuroMirror Team</p>
    </body>
    </html>
    """
    return send_email(email, subject, body, is_html=True)

def send_emergency_alert(user_email, username, alert_type, content):
    """Send emergency alert to therapist"""
    subject = f"URGENT: {alert_type.upper()} Alert - NeuroMirror"
    body = f"""
    <html>
    <body>
        <h2 style="color: red;">EMERGENCY ALERT</h2>
        <p><strong>Alert Type:</strong> {alert_type}</p>
        <p><strong>User Email:</strong> {user_email}</p>
        <p><strong>Username:</strong> {username}</p>
        <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Content:</strong></p>
        <div style="background-color: #f5f5f5; padding: 10px; border-left: 4px solid red;">
            {content}
        </div>
        <p>Please take appropriate action immediately.</p>
    </body>
    </html>
    """
    return send_email(THERAPIST_EMAIL, subject, body, is_html=True)

def detect_emergency_content(text):
    """Detect emergency content using Gemini API"""
    if not model:
        return False, "unknown"
    
    try:
        prompt = f"""
        
        You are a mental health safety AI.
        
        Analyze the user's message to detect signs of:
        - suicidal thoughts or intentions (e.g., wanting to die, overdose, not wake up)
        - self-harm (cutting, hurting oneself, harming one's body)
        - emotional crisis (panic, despair, hopelessness)
        - desire to harm others
        - illegal actions
        
        Respond ONLY in this JSON format:
        {{
          "is_emergency": true or false,
          "type": "suicidal" | "self_harm" | "crisis" | "harm_others" | "illegal" | "none",
          "confidence": float (0.0 - 1.0),
          "reason": "brief explanation of the detected emergency or risk"
        }}
        
        Example:
        Message: "I want to overdose and disappear"
        Response:
        {{
          "is_emergency": true,
          "type": "suicidal",
          "confidence": 0.97,
          "reason": "User expressed intent to overdose and die"
        }}
        
        Now analyze this message:
        "{text}"
        """
        response = model.generate_content(prompt)
        print("Gemini raw response:", response.text)

        try:
            raw = response.text.strip()
            # Remove markdown code block if present
            if raw.startswith("```") and raw.endswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])  # remove first and last lines
            
            result = json.loads(raw)            
            return result.get('is_emergency', False), result.get('type', 'unknown')
        except:
            # Fallback: simple keyword detection
            emergency_keywords = {
               'suicidal': [
                   'kill myself', 'end my life', 'suicide', 'want to die',
                   'better off dead', 'take my life', 'overdose', 'sleeping pills',
                   'hope i don\'t wake up', 'don’t want to wake up'
               ],
               'self_harm': [
                   'hurt myself', 'cut myself', 'self harm', 'burn myself',
                   'bleed', 'scratch until I bleed'
               ],
               'crisis': [
                   'can\'t take it anymore', 'mental breakdown', 'i give up',
                   'completely lost it', 'no hope left'
               ],
               'illegal': ['drugs', 'crime', 'illegal', 'stolen', 'weapon'],
               'harm_others': ['hurt someone', 'kill someone', 'shoot', 'attack']
            }           

            
            text_lower = text.lower()
            for emergency_type, keywords in emergency_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    print("1 :",emergency_type)
                    return True, emergency_type
            print("No emergency keywords matched.")
            return False, "none"
            
    except Exception as e:
        logger.error(f"Error detecting emergency content: {e}")
        return False, "unknown"

def get_user_from_session():
    """Get user from session token"""
    session_token = request.cookies.get('session_token')
    if not session_token:
        return None
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT u.id, u.username, u.email 
        FROM users u 
        JOIN user_sessions us ON u.id = us.user_id 
        WHERE us.session_token = ? AND us.expires_at > ?
    ''', (session_token, datetime.now().isoformat()))
    
    user = cursor.fetchone()
    conn.close()
    
    return user

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_user_from_session()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def get_therapeutic_suggestion(emotion, use_gemini=True):
    """Get therapeutic suggestion for detected emotion"""
    
    if use_gemini and model:
        try:
            prompt = f"""
            You are a compassionate AI therapist. A person is feeling {emotion}. 
            Provide a therapeutic response with:
            1. A supportive message (2-3 sentences)
            2. A practical tip for managing this emotion
            3. A suggested activity to help them feel better
            4. A type of calming sound that would help (e.g., "ocean waves", "forest sounds", "gentle rain")
            
            Be warm, empathetic, and professional. Focus on emotional regulation and self-care.
            Keep each section concise but meaningful.
            
            Format your response as JSON with keys: message, tip, activity, sound
            """
            
            response = model.generate_content(prompt)
            
            # Try to parse as JSON, fallback to structured text
            try:
                import json
                suggestion_data = json.loads(response.text)
                return suggestion_data
            except:
                # Fallback parsing if not JSON
                lines = response.text.split('\n')
                return {
                    "message": "You're doing great by acknowledging your feelings. Every emotion is valid and temporary.",
                    "tip": "Take slow, deep breaths and remind yourself that this feeling will pass.",
                    "activity": "Try a short mindfulness exercise or gentle movement.",
                    "sound": "calming nature sounds"
                }
                
        except Exception as e:
            logger.error(f"Error getting Gemini suggestion: {e}")
    
    # Fallback suggestions
    fallback_suggestions = {
        'happy': {
            "message": "Your positive energy is wonderful! Embrace this joyful moment and let it fill you with warmth.",
            "tip": "Share your happiness with others - positive emotions are contagious in the best way.",
            "activity": "Try dancing to your favorite song or call someone you care about.",
            "sound": "uplifting nature sounds"
        },
        'sad': {
            "message": "It's completely okay to feel this way. Your emotions are valid, and you're not alone in this.",
            "tip": "Allow yourself to feel without judgment. Sadness is a natural part of the human experience.",
            "activity": "Try gentle stretching, journaling, or listening to comforting music.",
            "sound": "gentle rain"
        },
        'angry': {
            "message": "Your feelings are valid. Let's channel this energy in a healthy, constructive way.",
            "tip": "Physical movement can help release tension. Take deep breaths and count to ten.",
            "activity": "Try a brief walk outside, some deep breathing, or write down your thoughts.",
            "sound": "flowing stream"
        },
        'fear': {
            "message": "You're braver than you feel right now. Fear is temporary, but your strength is lasting.",
            "tip": "Ground yourself by focusing on what you can control in this moment.",
            "activity": "Practice the 5-4-3-2-1 grounding technique: 5 things you see, 4 you hear, 3 you feel, 2 you smell, 1 you taste.",
            "sound": "peaceful forest"
        },
        'surprise': {
            "message": "Life is full of unexpected moments. You're handling this surprise with grace.",
            "tip": "Take a moment to process this new information. Surprises can lead to growth.",
            "activity": "Take a few mindful breaths and reflect on how you're feeling right now.",
            "sound": "gentle wind chimes"
        },
        'disgust': {
            "message": "Your boundaries and values are important. It's okay to feel this way about things that don't align with you.",
            "tip": "Distance yourself from what's bothering you if possible, and focus on what brings you peace.",
            "activity": "Engage in something that brings you joy or comfort - perhaps a hobby or time in nature.",
            "sound": "mountain breeze"
        },
        'neutral': {
            "message": "A balanced state is a gift. You're centered and ready for whatever comes your way.",
            "tip": "This is a perfect time for planning, reflection, or trying something new.",
            "activity": "Consider setting a small, achievable goal for today or practicing gratitude.",
            "sound": "ambient peace"
        }
    }
    
    return fallback_suggestions.get(emotion, fallback_suggestions['neutral'])

def get_chat_response(message, mood, chat_history, user=None):
    """Get therapeutic chat response using Gemini"""
    
    if not model:
        return "I'm here to listen and support you. Sometimes it helps just to know someone cares about you."
    
    try:
        # Check for emergency content first
        is_emergency, emergency_type = detect_emergency_content(message)
        
        if is_emergency and user:
            # Send emergency alert
            send_emergency_alert(user['email'], user['username'], emergency_type, message)
            
            # Store emergency alert in database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO emergency_alerts (user_id, user_email, username, alert_type, content)
                VALUES (?, ?, ?, ?, ?)
            ''', (user['id'], user['email'], user['username'], emergency_type, message))
            conn.commit()
            conn.close()
            
            # Return crisis response
            crisis_responses = {
                'suicidal': "I'm very concerned about what you're sharing. Your life has value and you're not alone. Please contact a mental health professional immediately or call your local crisis hotline. You can also reach out to our emergency contact at the bottom of this page.",
                'self_harm': "I hear that you're in a lot of pain right now. Please know that you don't have to face this alone. Reach out to a mental health professional or call a crisis hotline immediately. Your safety is the most important thing.",
                'crisis': "I can see you're going through an extremely difficult time. Please don't hesitate to reach out for immediate support. Contact a mental health professional or crisis hotline right away.",
                'illegal': "I understand you're sharing something serious. If you're involved in illegal activities, please consider speaking with a legal professional or counselor who can provide appropriate guidance.",
                'harm_others': "I'm concerned about what you've shared. If you're having thoughts of harming others, please seek immediate professional help. Your safety and the safety of others is important."
            }
            
            return crisis_responses.get(emergency_type, "I'm very concerned about what you're sharing. Please contact a mental health professional immediately for support.")
        
        # Build context from chat history
        context = ""
        if chat_history:
            context = "Previous conversation:\n"
            for msg in chat_history[-5:]:  # Last 5 messages for context
                sender = "Human" if msg.get('sender') == 'user' else "Therapist"
                context += f"{sender}: {msg.get('text', '')}\n"
        
        mood_context = f"The person's current detected mood is: {mood}" if mood else ""
        
        prompt = f"""
        You are a compassionate, professional AI therapist. You provide supportive, empathetic responses that help people process their emotions and find healthy coping strategies.
        
        {mood_context}
        
        {context}
        
        Human: {message}
        
        Guidelines for your response:
        - Be warm, empathetic, and non-judgmental
        - Acknowledge their feelings without minimizing them
        - Offer gentle guidance or coping strategies when appropriate
        - Ask thoughtful follow-up questions to encourage reflection
        - Keep responses concise but meaningful (2-4 sentences)
        - Use therapeutic techniques like validation, reframing, and mindfulness
        - Avoid giving medical advice or diagnosing
        - Focus on emotional support and self-care
        - If you detect signs of crisis, suicidal thoughts, or severe mental health issues, encourage them to contact a mental health professional immediately (use indian hepline).
        
        Respond as a caring therapist:
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error getting chat response: {e}")
        return "I'm here for you. Your feelings are valid, and it's okay to take things one step at a time."

@app.route('/auth/signup', methods=['POST'])
def signup():
    """User registration endpoint"""
    try:
        data = request.get_json()
        
        if not data or not all(k in data for k in ['username', 'email', 'password']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        
        # Validate input
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Check if user already exists
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        existing_user = cursor.fetchone()
        
        if existing_user:
            conn.close()
            return jsonify({'error': 'Username or email already exists'}), 409
        
        # Create new user
        password_hash = hash_password(password)
        verification_token = secrets.token_urlsafe(32)
        
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, verification_token)
            VALUES (?, ?, ?, ?)
        ''', (username, email, password_hash, verification_token))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Send welcome email
        send_welcome_email(email, username)
        
        return jsonify({
            'message': 'Account created successfully. Please check your email for verification.',
            'user_id': user_id
        }), 201
        
    except Exception as e:
        logger.error(f"Error in signup: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        
        if not data or not all(k in data for k in ['username', 'password']):
            return jsonify({'error': 'Missing username or password'}), 400
        
        username = data['username'].strip()
        password = data['password']
        
        # Find user by username or email
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, email, password_hash FROM users WHERE username = ? OR email = ?', (username, username))
        user = cursor.fetchone()
        
        if not user or user['password_hash'] != hash_password(password):
            conn.close()
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Create session
        session_token = generate_session_token()
        expires_at = datetime.now() + timedelta(days=30)
        
        cursor.execute('''
            INSERT INTO user_sessions (user_id, session_token, expires_at)
            VALUES (?, ?, ?)
        ''', (user['id'], session_token, expires_at.isoformat()))
        
        # Update last login
        cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.now().isoformat(), user['id']))
        
        conn.commit()
        conn.close()
        
        response = jsonify({
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email']
            }
        })
        
        # Set session cookie
        response.set_cookie('session_token', session_token, max_age=30*24*60*60, httponly=True, secure=False, samesite='Lax')
        
        return response
        
    except Exception as e:
        logger.error(f"Error in login: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/auth/logout', methods=['POST'])
def logout():
    """User logout endpoint"""
    try:
        session_token = request.cookies.get('session_token')
        if session_token:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM user_sessions WHERE session_token = ?', (session_token,))
            conn.commit()
            conn.close()
        
        response = jsonify({'message': 'Logout successful'})
        response.delete_cookie('session_token')
        return response
        
    except Exception as e:
        logger.error(f"Error in logout: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/auth/me', methods=['GET'])
def get_current_user():
    """Get current user information"""
    try:
        user = get_user_from_session()
        if not user:
            return jsonify({'error': 'Not authenticated'}), 401
        
        return jsonify({
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email']
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/detect', methods=['POST'])
@require_auth
def detect_mood():
    """Detect emotion from uploaded image using DeepFace"""
    try:
        user = get_user_from_session()
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode the image
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        logger.info(f"Processing image for user {user['username']} - Shape: {image.shape}")

        # Detect emotion using DeepFace
        emotion, confidence, all_emotions = detect_emotion_deepface(image)
        
        if emotion is None:
            return jsonify({'error': 'Emotion detection failed'}), 400
        
        # Store in database with user_id
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO mood_history (user_id, emotion, confidence, session_id, all_emotions) VALUES (?, ?, ?, ?, ?)',
            (user['id'], emotion, confidence, data.get('session_id', 'default'), json.dumps(all_emotions))
        )
        conn.commit()
        conn.close()
        
        # Get therapeutic suggestion
        suggestion = get_therapeutic_suggestion(emotion)
        
        # Store suggestion in database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO suggestions (emotion, message, tip, activity, sound) VALUES (?, ?, ?, ?, ?)',
            (emotion, suggestion['message'], suggestion['tip'], 
             suggestion.get('activity', ''), suggestion.get('sound', ''))
        )
        conn.commit()
        conn.close()
        
        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence, 2),
            'all_emotions': all_emotions,
            'message': suggestion['message'],
            'tip': suggestion['tip'],
            'activity': suggestion.get('activity', ''),
            'sound': suggestion.get('sound', '')
        })
        
    except Exception as e:
        logger.error(f"Error in detect_mood: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/chat', methods=['POST'])
@require_auth
def chat():
    """Handle therapeutic chat conversation"""
    try:
        user = get_user_from_session()
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_message = data['message']
        mood = data.get('mood')
        chat_history = data.get('chat_history', [])
        session_id = data.get('session_id', 'default')
        
        # Store user message
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO chat_history (user_id, message, sender, mood, session_id) VALUES (?, ?, ?, ?, ?)',
            (user['id'], user_message, 'user', mood, session_id)
        )
        
        # Get AI response with user context for emergency detection
        ai_response = get_chat_response(user_message, mood, chat_history, user)
        
        # Store AI response
        cursor.execute(
            'INSERT INTO chat_history (user_id, message, sender, mood, session_id) VALUES (?, ?, ?, ?, ?)',
            (user['id'], ai_response, 'therapist', mood, session_id)
        )
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'response': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/mood-history', methods=['GET'])
@require_auth
def get_mood_history():
    """Get mood detection history"""
    try:
        user = get_user_from_session()
        session_id = request.args.get('session_id', 'default')
        limit = request.args.get('limit', 50, type=int)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT emotion, confidence, timestamp, all_emotions FROM mood_history WHERE user_id = ? AND session_id = ? ORDER BY timestamp DESC LIMIT ?',
            (user['id'], session_id, limit)
        )
        
        history = []
        for row in cursor.fetchall():
            all_emotions = {}
            try:
                if row['all_emotions']:
                    all_emotions = json.loads(row['all_emotions'])
            except:
                pass
                
            history.append({
                'emotion': row['emotion'],
                'confidence': row['confidence'],
                'timestamp': row['timestamp'],
                'all_emotions': all_emotions
            })
        
        conn.close()
        return jsonify({'history': history})
        
    except Exception as e:
        logger.error(f"Error getting mood history: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/chat-history', methods=['GET'])
@require_auth
def get_chat_history():
    """Get chat conversation history"""
    try:
        user = get_user_from_session()
        session_id = request.args.get('session_id', 'default')
        limit = request.args.get('limit', 100, type=int)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT message, sender, mood, timestamp FROM chat_history WHERE user_id = ? AND session_id = ? ORDER BY timestamp ASC LIMIT ?',
            (user['id'], session_id, limit)
        )
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'message': row['message'],
                'sender': row['sender'],
                'mood': row['mood'],
                'timestamp': row['timestamp']
            })
        
        conn.close()
        return jsonify({'history': history})
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/user/mood-journey', methods=['GET'])
@require_auth
def get_user_mood_journey():
    """Get user's complete mood journey data"""
    try:
        user = get_user_from_session()
        days = request.args.get('days', 30, type=int)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get mood history for the specified days
        cursor.execute('''
            SELECT emotion, confidence, timestamp, all_emotions 
            FROM mood_history 
            WHERE user_id = ? AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp ASC
        '''.format(days), (user['id'],))
        
        mood_data = []
        for row in cursor.fetchall():
            all_emotions = {}
            try:
                if row['all_emotions']:
                    all_emotions = json.loads(row['all_emotions'])
            except:
                pass
                
            mood_data.append({
                'emotion': row['emotion'],
                'confidence': row['confidence'],
                'timestamp': row['timestamp'],
                'all_emotions': all_emotions
            })
        
        # Get emotion frequency
        cursor.execute('''
            SELECT emotion, COUNT(*) as count 
            FROM mood_history 
            WHERE user_id = ? AND timestamp >= datetime('now', '-{} days')
            GROUP BY emotion 
            ORDER BY count DESC
        '''.format(days), (user['id'],))
        
        emotion_frequency = {}
        for row in cursor.fetchall():
            emotion_frequency[row['emotion']] = row['count']
        
        conn.close()
        
        return jsonify({
            'mood_data': mood_data,
            'emotion_frequency': emotion_frequency,
            'total_entries': len(mood_data)
        })
        
    except Exception as e:
        logger.error(f"Error getting user mood journey: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/emergency-contact', methods=['GET'])
def get_emergency_contact():
    """Get emergency contact information"""
    return jsonify({
        'emergency_contact': {
            'name': 'Dr. Sarah Johnson',
            'email': 'dr.sarah.johnson@example.com',
            'phone': '+1-555-0123',
            'specialization': 'Licensed Clinical Psychologist',
            'availability': '24/7 Crisis Support',
            'message': 'If you are experiencing a mental health crisis, please contact me immediately. Your safety and well-being are my top priority.'
        }
    })

@app.route('/debug/test-deepface', methods=['POST'])
@require_auth
def debug_test_deepface():
    """Debug endpoint to test DeepFace emotion detection"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode the image
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        logger.info(f"Debug: Testing DeepFace with image shape: {image.shape}")
        
        # Test DeepFace emotion detection
        emotion, confidence, all_emotions = detect_emotion_deepface(image)
        
        # Test different backends
        backends_results = {}
        test_backends = ['opencv', 'mtcnn', 'retinaface', 'mediapipe', 'ssd']
        
        for backend in test_backends:
            try:
                result = DeepFace.analyze(
                    img_path=image,
                    actions=['emotion'],
                    detector_backend=backend,
                    enforce_detection=False,
                    silent=True
                )
                
                if isinstance(result, list):
                    emotion_data = result[0]['emotion'] if result else {}
                else:
                    emotion_data = result['emotion']
                
                if emotion_data:
                    dominant = max(emotion_data, key=emotion_data.get)
                    backends_results[backend] = {
                        'success': True,
                        'dominant_emotion': dominant,
                        'confidence': emotion_data[dominant],
                        'all_emotions': emotion_data
                    }
                else:
                    backends_results[backend] = {'success': False, 'error': 'No emotion data'}
                    
            except Exception as e:
                backends_results[backend] = {'success': False, 'error': str(e)}
        
        return jsonify({
            'main_result': {
                'emotion': emotion,
                'confidence': confidence,
                'all_emotions': all_emotions
            },
            'backends_test': backends_results,
            'image_info': {
                'shape': image.shape,
                'dtype': str(image.dtype),
                'min_val': float(image.min()),
                'max_val': float(image.max())
            }
        })
        
    except Exception as e:
        logger.error(f"Error in DeepFace debug test: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-status', methods=['GET'])
def get_model_status():
    """Get emotion detection model status"""
    
    # Test DeepFace availability
    deepface_status = {
        'available': True,
        'error': None
    }
    
    try:
        # Try to import and test DeepFace
        import deepface
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_result = DeepFace.analyze(
            img_path=dummy_image,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=False,
            silent=True
        )
        deepface_status['test_successful'] = True
    except Exception as e:
        deepface_status['available'] = False
        deepface_status['error'] = str(e)
        deepface_status['test_successful'] = False
    
    # Available backends
    available_backends = []
    test_backends = ['opencv', 'mtcnn', 'retinaface', 'mediapipe', 'ssd']
    
    for backend in test_backends:
        try:
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(
                img_path=dummy_image,
                actions=['emotion'],
                detector_backend=backend,
                enforce_detection=False,
                silent=True
            )
            available_backends.append(backend)
        except:
            pass
    
    status = {
        'emotion_detection_method': 'DeepFace',
        'deepface_status': deepface_status,
        'available_backends': available_backends,
        'default_backend': DEEPFACE_BACKEND,
        'emotion_mapping': EMOTION_MAPPING,
        'opencv_version': cv2.__version__,
        'initialization_success': deepface_init_success,
        'components': {
            'deepface': deepface_status['available'],
            'opencv': True,
            'gemini_api': model is not None
        }
    }
    
    return jsonify(status)

@app.route('/check-face', methods=['POST'])
def check_face_presence():
    """Lightweight check just for face presence"""
    try:
        data = request.get_json()
        image = decode_base64_image(data['image'])
        
        # Use try-catch to handle face detection gracefully
        try:
            # First try with enforce_detection=True to get accurate detection
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=DEEPFACE_BACKEND,
                enforce_detection=True
            )
            # If we reach here, face(s) were detected
            return jsonify({
                'face_detected': True,
                'face_count': len(faces)
            })
            
        except ValueError as face_error:
            # This happens when no face is detected with enforce_detection=True
            if "Face could not be detected" in str(face_error):
                return jsonify({
                    'face_detected': False,
                    'face_count': 0
                })
            else:
                # Re-raise if it's a different ValueError
                raise face_error
                
        except Exception as face_error:
            # Fallback: try with enforce_detection=False
            print(f"Primary face detection failed: {face_error}")
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=DEEPFACE_BACKEND,
                enforce_detection=False
            )
            
            # With enforce_detection=False, we need to check if valid faces were extracted
            valid_faces = []
            for face in faces:
                # Check if the face array has reasonable dimensions and isn't just noise
                if face.shape[0] > 10 and face.shape[1] > 10:  # Minimum face size
                    # Check if the face isn't just a black/empty image
                    if face.mean() > 0.05:  # Basic check for non-empty face
                        valid_faces.append(face)
            
            return jsonify({
                'face_detected': len(valid_faces) > 0,
                'face_count': len(valid_faces)
            })
        
    except Exception as e:
        print(f"Face check error: {str(e)}")
        return jsonify({
            'error': str(e),
            'face_detected': False,
            'face_count': 0
        }), 500
    
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'gemini_configured': model is not None,
        'emotion_detection_ready': deepface_init_success,
        'emotion_detection_method': 'DeepFace',
        'components': {
            'database': True,
            'deepface': deepface_init_success,
            'gemini_api': model is not None
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

    