from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import torch
from torchvision import models, transforms
from PIL import Image
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import torch.nn as nn
import json
import re
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np
import io
import base64
from flask_sqlalchemy import SQLAlchemy
import requests
import time
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'pool_recycle': 3600,
    'pool_pre_ping': True,
    'pool_use_lifo': True
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Groq API configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Fixed API endpoint
GROQ_MODEL = "llama-3.3-70b-versatile"  # Using a supported Groq model

# System prompt for the AI
SYSTEM_PROMPT = """You are an advanced AI assistant specializing in blood group detection through fingerprint analysis and general medical knowledge. Your capabilities include:

Core Knowledge:
- Comprehensive understanding of blood types (A, B, AB, O) and Rh factors
- Detailed knowledge of fingerprint patterns (arch, loop, whorl) and their characteristics
- Understanding of the correlation between blood groups and fingerprint patterns
- Medical terminology and concepts related to hematology
- AI and machine learning technology used in medical analysis

Keep responses concise, professional, and focused on medical and technical accuracy.
If you're not sure about something, acknowledge the uncertainty.
Do not provide medical advice or diagnosis."""

# Chat history with enhanced context management
chat_histories = {}

def get_chat_history(session_id):
    """Get chat history for a session with enhanced context management"""
    if session_id not in chat_histories:
        chat_histories[session_id] = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
            "last_interaction": datetime.now(),
            "context": {
                "topics_discussed": set(),
                "user_interests": set(),
                "technical_level": "medium"  # default level
            }
        }
    return chat_histories[session_id]

def update_context(session_id, user_message, ai_response):
    """Update context based on conversation"""
    if session_id in chat_histories:
        context = chat_histories[session_id]["context"]
        
        # Update last interaction time
        chat_histories[session_id]["last_interaction"] = datetime.now()
        
        # Extract potential topics of interest
        medical_terms = ["blood", "fingerprint", "pattern", "type", "analysis", "rh", "factor"]
        technical_terms = ["ai", "machine learning", "algorithm", "detection", "accuracy"]
        
        msg_lower = user_message.lower()
        
        # Update topics discussed
        for term in medical_terms + technical_terms:
            if term in msg_lower:
                context["topics_discussed"].add(term)
        
        # Adjust technical level based on user's language
        technical_indicators = len([term for term in technical_terms if term in msg_lower])
        if technical_indicators > 2:
            context["technical_level"] = "high"
        elif technical_indicators > 0:
            context["technical_level"] = "medium"

def validate_api_key():
    """Validate the Groq API key by making a test request"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        test_data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7
        }
        response = requests.post(GROQ_API_URL, headers=headers, json=test_data, timeout=10)
        
        if response.status_code == 400:
            error_message = "API Response: "
            if response.text:
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_message += str(error_data['error'])
                    else:
                        error_message += response.text
                except:
                    error_message += response.text
            logger.error(error_message)
            
        response.raise_for_status()
        logger.info("API key validation successful")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"API key validation failed: {str(e)}")
        if response := getattr(e, 'response', None):
            logger.error(f"Response status: {response.status_code}")
            logger.error(f"Response body: {response.text}")
        return False

# Validate API key on startup
if not validate_api_key():
    logger.error("Failed to validate Groq API key. Please check your API key and try again.")

def generate_ai_response(user_message, session_id):
    """Generate enhanced AI response using Groq API"""
    try:
        logger.info(f"Generating response for message: {user_message}")
        
        # Get chat history
        chat_data = get_chat_history(session_id)
        messages = chat_data["messages"]
        
        # Add user message to history
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare the request to Groq API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Complete request data with supported parameters
        data = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9
        }
        
        logger.debug(f"Sending request to Groq API with data: {json.dumps(data, indent=2)}")
        
        # Make request to Groq API
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        
        # Log the response for debugging
        logger.debug(f"Groq API response status: {response.status_code}")
        logger.debug(f"Groq API response: {response.text}")
        
        if response.status_code == 200:
            response_data = response.json()
            ai_response = response_data['choices'][0]['message']['content']
            
            # Add AI response to history
            messages.append({
                "role": "assistant",
                "content": ai_response
            })
            
            # Trim history if it gets too long
            if len(messages) > 11:  # system prompt + 10 messages
                messages = [messages[0]] + messages[-10:]
            
            return ai_response
        else:
            error_message = f"API request failed with status {response.status_code}"
            if response.text:
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_message += f": {error_data['error']}"
                except:
                    error_message += f": {response.text}"
            logger.error(error_message)
            return "I apologize, but I'm having trouble processing your request. Please try again."
            
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return "I apologize, but I'm having trouble connecting to my knowledge base. Please try again in a moment."

# DB Setup
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, fullname TEXT, email TEXT UNIQUE, username TEXT UNIQUE, password TEXT)'
        )
        conn.commit()

init_db()

# Model Setup: ResNet18 modified for 8 blood groups
num_classes = 8
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('fingerprint_blood_group_model.pth', map_location=torch.device('cpu')))
model.eval()

# Transformation (same as during training)
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

blood_groups = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Enhanced knowledge base
KNOWLEDGE_BASE = {
    "blood_groups": {
        "general": "Blood groups are classifications of blood based on the presence or absence of antigens on the surface of red blood cells. The two main classification systems are ABO and Rh factor.",
        "types": {
            "A": "Type A has A antigens on red blood cells and B antibodies in plasma",
            "B": "Type B has B antigens on red blood cells and A antibodies in plasma",
            "AB": "Type AB has both A and B antigens on red blood cells and no antibodies",
            "O": "Type O has no antigens on red blood cells but both A and B antibodies in plasma"
        },
        "rh_factor": {
            "positive": "Rh-positive means RBC have the Rh protein",
            "negative": "Rh-negative means RBC lack the Rh protein",
            "inheritance": "Rh factor is inherited from parents, with Rh+ being dominant"
        },
        "inheritance": {
            "patterns": "Blood type is inherited through multiple alleles: A, B, and O. Each person inherits one allele from each parent.",
            "dominance": "A and B are codominant, while O is recessive",
            "combinations": {
                "A_parent": "If one parent is type A: Child could be A or O depending on the other parent",
                "B_parent": "If one parent is type B: Child could be B or O depending on the other parent",
                "AB_parent": "If one parent is type AB: Child cannot be type O",
                "O_parent": "If one parent is type O: Child will inherit O from that parent"
            }
        }
    },
    "fingerprint": {
        "patterns": {
            "arch": "Simple curved patterns without deltas",
            "loop": "Pattern that recurves and passes out the same side it entered",
            "whorl": "Circular or spiral patterns with two deltas",
            "composite": "Combination of two or more basic patterns"
        },
        "characteristics": {
            "ridges": "Raised portions of skin on fingertips forming unique patterns",
            "minutiae": "Specific points where ridge patterns change, like endings or bifurcations",
            "core": "Center point of the fingerprint pattern",
            "delta": "Triangular meeting point of ridge patterns"
        },
        "analysis": {
            "imaging": "High-resolution scanning captures detailed ridge patterns",
            "processing": "Advanced image processing enhances pattern clarity",
            "feature_extraction": "AI identifies unique characteristics and patterns",
            "correlation": "Machine learning models analyze pattern distribution"
        }
    },
    "technology": {
        "ai_model": {
            "description": "We use a state-of-the-art Convolutional Neural Network (CNN) based on ResNet18 architecture",
            "features": "The model analyzes complex fingerprint patterns and correlates them with blood groups",
            "training": "Trained on a large dataset of fingerprint images with verified blood groups",
            "accuracy": "Shows promising results in initial studies, with continuous improvement through learning"
        },
        "process": {
            "steps": [
                "Upload clear fingerprint image",
                "AI enhances and processes the image",
                "Model analyzes fingerprint patterns",
                "Blood group prediction is generated",
                "Results are cross-referenced with genetic patterns"
            ],
            "requirements": {
                "image": "Clear, high-resolution fingerprint scan or photo",
                "quality": "Clean, undamaged fingerprint surface",
                "position": "Centered, complete fingerprint impression"
            }
        },
        "privacy": {
            "data_handling": "All images are processed securely and not stored",
            "encryption": "Data transmission uses industry-standard encryption",
            "compliance": "Adheres to data protection regulations"
        }
    }
}

def generate_detailed_response(user_message):
    """Generate natural and detailed responses based on user input"""
    message = user_message.lower()
    
    # Initialize response components
    response_parts = []
    
    # Helper function to search knowledge base
    def search_dict(d, search_term, path=[]):
        results = []
        for k, v in d.items():
            new_path = path + [k]
            if isinstance(v, dict):
                results.extend(search_dict(v, search_term, new_path))
            elif isinstance(v, str) and search_term in v.lower():
                results.append((new_path, v))
            elif isinstance(v, list) and any(search_term in x.lower() for x in v):
                results.append((new_path, '. '.join(v)))
        return results
    
    # Process questions about blood groups
    if "blood" in message or "group" in message:
        if "inherit" in message:
            response_parts.append(KNOWLEDGE_BASE["blood_groups"]["inheritance"]["patterns"])
            response_parts.append(KNOWLEDGE_BASE["blood_groups"]["inheritance"]["dominance"])
        elif "type" in message or "different" in message:
            response_parts.append(KNOWLEDGE_BASE["blood_groups"]["general"])
            for blood_type, desc in KNOWLEDGE_BASE["blood_groups"]["types"].items():
                response_parts.append(f"{blood_type}: {desc}")
        elif "rh" in message or "positive" in message or "negative" in message:
            response_parts.append(KNOWLEDGE_BASE["blood_groups"]["rh_factor"]["inheritance"])
            response_parts.append(KNOWLEDGE_BASE["blood_groups"]["rh_factor"]["positive"])
            response_parts.append(KNOWLEDGE_BASE["blood_groups"]["rh_factor"]["negative"])
    
    # Process questions about fingerprints
    elif "fingerprint" in message:
        if "pattern" in message:
            response_parts.append("Fingerprint patterns include:")
            for pattern, desc in KNOWLEDGE_BASE["fingerprint"]["patterns"].items():
                response_parts.append(f"- {pattern.capitalize()}: {desc}")
        elif "characteris" in message or "feature" in message:
            response_parts.append("Key fingerprint characteristics include:")
            for char, desc in KNOWLEDGE_BASE["fingerprint"]["characteristics"].items():
                response_parts.append(f"- {char.capitalize()}: {desc}")
        elif "analy" in message or "process" in message:
            response_parts.append("The fingerprint analysis process involves:")
            for step, desc in KNOWLEDGE_BASE["fingerprint"]["analysis"].items():
                response_parts.append(f"- {step.replace('_', ' ').capitalize()}: {desc}")
    
    # Process questions about technology
    elif "technology" in message or "how" in message or "work" in message:
        if "model" in message or "ai" in message:
            response_parts.append(KNOWLEDGE_BASE["technology"]["ai_model"]["description"])
            response_parts.append(KNOWLEDGE_BASE["technology"]["ai_model"]["features"])
        elif "process" in message or "step" in message:
            response_parts.append("The process involves these steps:")
            response_parts.extend(KNOWLEDGE_BASE["technology"]["process"]["steps"])
        elif "privacy" in message or "secure" in message:
            for key, value in KNOWLEDGE_BASE["technology"]["privacy"].items():
                response_parts.append(value)
    
    # Process questions about accuracy or reliability
    elif "accura" in message or "reliable" in message:
        response_parts.append(KNOWLEDGE_BASE["technology"]["ai_model"]["accuracy"])
        response_parts.append("For best results, ensure:")
        for req, desc in KNOWLEDGE_BASE["technology"]["process"]["requirements"].items():
            response_parts.append(f"- {desc}")
    
    # If no specific matches, search entire knowledge base
    if not response_parts:
        all_results = []
        for section in KNOWLEDGE_BASE.keys():
            results = search_dict(KNOWLEDGE_BASE[section], message)
            all_results.extend(results)
        
        if all_results:
            # Take the most relevant results
            for path, content in all_results[:2]:
                response_parts.append(content)
        else:
            response_parts.append(
                "I understand you're asking about blood group prediction through fingerprint analysis. "
                "Would you like to know about:\n"
                "- The technology and how it works\n"
                "- Blood groups and inheritance\n"
                "- Fingerprint patterns and analysis\n"
                "- The accuracy and requirements"
            )
    
    # Add educational follow-up if response is short
    if len(response_parts) == 1:
        response_parts.append("\nWould you like to know more about:")
        if "blood" not in message.lower():
            response_parts.append("- Blood group inheritance patterns")
        if "fingerprint" not in message.lower():
            response_parts.append("- Fingerprint pattern types and analysis")
        if "technology" not in message.lower():
            response_parts.append("- How our AI technology works")
    
    return "\n".join(response_parts)

@app.route('/')
def home():
    return render_template('hospitalhomepage.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            flash('Login successful!', 'success')
            return redirect(url_for('predict_blood_group'))
        else:
            flash('Invalid username or password!', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']

        if password != confirmpassword:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (fullname, email, username, password) VALUES (?, ?, ?, ?)',
                         (fullname, email, username, hashed_password))
            conn.commit()
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
        finally:
            conn.close()
    return render_template('signup.html')

@app.route('/predictor')
def predictor():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('predictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    father_bg = request.form.get('father_blood_group', '').strip().upper()
    mother_bg = request.form.get('mother_blood_group', '').strip().upper()
    file = request.files.get('file')

    blood_groups = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O-', 'O+']

    def get_abo(bg):
        return bg[:-1]

    def get_rh(bg):
        return bg[-1]

    def calculate_possible_bgs(f_bg, m_bg):
        f_abo = get_abo(f_bg)
        m_abo = get_abo(m_bg)
        
        # COMMON REAL-LIFE OUTCOMES
        common_abo = []
        
        # When one parent is O
        if 'O' in {f_abo, m_abo}:
            other = 'A' if 'A' in {f_abo, m_abo} else 'B' if 'B' in {f_abo, m_abo} else 'O'
            if other == 'A': common_abo = ['A', 'O']
            elif other == 'B': common_abo = ['B', 'O']
            else: common_abo = ['O']
        
        # When one parent is AB
        elif 'AB' in {f_abo, m_abo}:
            other = 'A' if 'A' in {f_abo, m_abo} else 'B' if 'B' in {f_abo, m_abo} else 'AB'
            if other == 'A': common_abo = ['A', 'B']
            elif other == 'B': common_abo = ['A', 'B']
            else: common_abo = ['A', 'B', 'AB']
        
        # Other cases
        elif f_abo == 'A' and m_abo == 'A':
            common_abo = ['A']
        elif f_abo == 'B' and m_abo == 'B':
            common_abo = ['B']
        elif {f_abo, m_abo} == {'A', 'B'}:
            common_abo = ['A', 'B']
        
        # Rh factor - always show + if parents are +
        f_rh = get_rh(f_bg)
        m_rh = get_rh(m_bg)
        if f_rh == '+' and m_rh == '+':
            common_rh = ['+']  # Force + when both parents are +
        else:
            common_rh = ['+', '-']
        
        return sorted([f"{abo}{rh}" for abo in common_abo for rh in common_rh])

    # Validate parents' blood groups
    parents_valid = father_bg in blood_groups and mother_bg in blood_groups
    possible_bgs = calculate_possible_bgs(father_bg, mother_bg) if parents_valid else []

    if not file or file.filename == '':
        if parents_valid:
            return render_template('result.html',
                                blood_group=", ".join(possible_bgs),
                                possible_groups=possible_bgs,
                                message="Most likely blood groups from parents:",
                                has_image=False)
        else:
            flash("Please provide valid blood groups for both parents or upload an image.", "error")
            return redirect(url_for('predict_blood_group'))

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        img = Image.open(file_path).convert('RGB')
        img_tensor = data_transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_bg = blood_groups[predicted.item()]

        # Fix Rh discrepancy when parents are both +
        if parents_valid and (get_rh(father_bg) == '+' and get_rh(mother_bg) == '+'):
            if get_rh(predicted_bg) == '-':
                predicted_bg = predicted_bg[:-1] + '+'  # Change - to +

        if parents_valid:
            if predicted_bg in possible_bgs:
                message = f"Model prediction matches likely parental combinations: {predicted_bg}"
            else:
                # Show both prediction and likely parental combinations
                message = f"Model predicted: {predicted_bg} | Likely from parents: {', '.join(possible_bgs)}"
            final_pred = predicted_bg  # Always show model prediction, never None
        else:
            message = f"Prediction based on fingerprint: {predicted_bg}"
            final_pred = predicted_bg

        return render_template('result.html',
                            blood_group=final_pred,
                            possible_groups=possible_bgs if parents_valid else [predicted_bg],
                            message=message,
                            has_image=True)

    except Exception as e:
        flash(f"Error processing image: {str(e)}", "error")
        return redirect(url_for('predict_blood_group'))
    
@app.route('/predict_blood_group')
def predict_blood_group():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    return render_template('predictor.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Blood group inheritance rules
BLOOD_GROUP_INHERITANCE = {
    "A": {
        "A": ["A", "O"],
        "B": ["A", "B", "AB", "O"],
        "AB": ["A", "B", "AB"],
        "O": ["A", "O"]
    },
    "B": {
        "A": ["A", "B", "AB", "O"],
        "B": ["B", "O"],
        "AB": ["A", "B", "AB"],
        "O": ["B", "O"]
    },
    "AB": {
        "A": ["A", "B", "AB"],
        "B": ["A", "B", "AB"],
        "AB": ["A", "B", "AB"],
        "O": ["A", "B"]
    },
    "O": {
        "A": ["A", "O"],
        "B": ["B", "O"],
        "AB": ["A", "B"],
        "O": ["O"]
    }
}

RH_INHERITANCE = {
    "+": {
        "+": ["+"],
        "-": ["+", "-"]
    },
    "-": {
        "+": ["+", "-"],
        "-": ["-"]
    }
}

def get_possible_blood_groups(mother_bg, father_bg):
    """Calculate possible blood groups based on parents' blood groups"""
    mother_abo = ''.join(c for c in mother_bg if c not in '+-')
    father_abo = ''.join(c for c in father_bg if c not in '+-')
    mother_rh = '+' if '+' in mother_bg else '-'
    father_rh = '+' if '+' in father_bg else '-'

    possible_abo = BLOOD_GROUP_INHERITANCE[mother_abo][father_abo]
    possible_rh = RH_INHERITANCE[mother_rh][father_rh]

    return [f"{abo}{rh}" for abo in possible_abo for rh in possible_rh]

def validate_blood_group(blood_group):
    """Validate blood group format"""
    valid_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    return blood_group in valid_groups

def process_fingerprint_image(image_data):
    """Process fingerprint image and return prediction"""
    try:
        # Convert image to RGB if needed
        if image_data.mode != 'RGB':
            image_data = image_data.convert('RGB')
        
        # Apply the same transforms as during training
        image_tensor = data_transform(image_data).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
            
        return blood_groups[predicted.item()], confidence
    except Exception as e:
        return None, 0.0

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        user_message = data.get('message', '')
        is_speech = data.get('is_speech', False)
        
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        logger.info(f"Received chat request with message: {user_message} (Speech: {is_speech})")
        
        # Get or create session ID
        session_id = session.get('chat_session_id')
        if not session_id:
            session_id = os.urandom(16).hex()
            session['chat_session_id'] = session_id
        
        try:
            # Generate AI response with timeout
            response = generate_ai_response(user_message, session_id)
            
            # Get chat history for context-aware suggestions
            chat_data = get_chat_history(session_id)
            context = chat_data["context"]
            
            # Base suggestions
            suggestions = [
                "Tell me about blood types",
                "How do fingerprint patterns work?",
                "What's the accuracy of detection?",
                "How is my data protected?"
            ]
            
            # Add speech-specific suggestions if the input was speech
            if is_speech:
                speech_suggestions = [
                    "Can you explain the process again?",
                    "What are the different blood types?",
                    "How does the AI work?",
                    "Is this method accurate?",
                    "Can you speak more slowly?",
                    "Could you repeat that?"
                ]
                suggestions.extend(speech_suggestions)
            
            return jsonify({
                "response": response,
                "timestamp": datetime.now().strftime("%H:%M"),
                "suggestions": suggestions[:8],  # Limit to 8 suggestions
                "is_speech": is_speech
            })
            
        except requests.exceptions.Timeout:
            logger.error("Request to AI service timed out")
            return jsonify({
                "error": "The request timed out. Please try again.",
                "is_timeout": True
            }), 504
            
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to AI service")
            return jsonify({
                "error": "Could not connect to the AI service. Please check your internet connection.",
                "is_connection_error": True
            }), 503
            
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return jsonify({
                "error": "An error occurred while processing your request. Please try again.",
                "is_error": True
            }), 500
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": "An unexpected error occurred. Please try again.",
            "is_error": True
        }), 500

@app.route('/api/reset_chat', methods=['POST'])
def reset_chat():
    """Reset the chat history for the current session"""
    session_id = session.get('chat_session_id')
    if session_id and session_id in chat_histories:
        chat_histories[session_id] = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
            "last_interaction": datetime.now(),
            "context": {
                "topics_discussed": set(),
                "user_interests": set(),
                "technical_level": "medium"
            }
        }
    return jsonify({"status": "success"})

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a JPG or PNG image.'}), 400
    
    try:
        # Read and validate image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Process image and get prediction
        predicted_group, confidence = process_fingerprint_image(image)
        
        if predicted_group:
            return jsonify({
                'success': True,
                'predicted_group': predicted_group,
                'confidence': confidence
            })
        else:
            return jsonify({'error': 'Could not process the image properly'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    """Handle speech-to-text conversion requests"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Here you would typically process the audio file
        # For now, we'll return a mock response since actual speech processing
        # would require additional libraries and setup
        
        return jsonify({
            "success": True,
            "message": "Speech processing endpoint ready",
            "note": "This is a placeholder endpoint. Actual speech processing would require additional setup."
        })
        
    except Exception as e:
        logger.error(f"Error in speech-to-text endpoint: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

# Add error handlers
@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('error.html', error="Internal Server Error"), 500

@app.errorhandler(502)
def bad_gateway(error):
    return render_template('error.html', error="Bad Gateway"), 502

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error="Page Not Found"), 404

# Add before_request handler for database connection
@app.before_request
def before_request():
    try:
        db.session.execute('SELECT 1')
    except Exception as e:
        db.session.rollback()
        raise

# Add after_request handler
@app.after_request
def after_request(response):
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)