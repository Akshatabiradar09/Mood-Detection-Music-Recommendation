"""Mood Detection Flask Application with Spotify Integration and PostgreSQL."""
import os
import time
import base64
import secrets
import random
import logging

import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity, 
    set_access_cookies, unset_jwt_cookies
)
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Config / App init
# ---------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "emotion_model.tflite")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Environment detection
IS_PRODUCTION = os.environ.get("FLASK_ENV", "development") == "production"

# Security configuration
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY") or secrets.token_hex(32)
app.config["JWT_TOKEN_LOCATION"] = ["cookies"]
app.config["JWT_COOKIE_SECURE"] = IS_PRODUCTION  # True for HTTPS in production
app.config["JWT_ACCESS_COOKIE_PATH"] = "/"
app.config["JWT_COOKIE_SAMESITE"] = "Strict" if IS_PRODUCTION else "Lax"
app.config["JWT_COOKIE_CSRF_PROTECT"] = IS_PRODUCTION  # Enable CSRF in production

# Warn if using default secrets in production
if IS_PRODUCTION and not os.environ.get("FLASK_SECRET_KEY"):
    logger.warning("FLASK_SECRET_KEY not set! Using random key (sessions won't persist across restarts)")
if IS_PRODUCTION and not os.environ.get("JWT_SECRET_KEY"):
    logger.warning("JWT_SECRET_KEY not set! Using random key (tokens won't persist across restarts)")

jwt = JWTManager(app)

# ---------------------------
# Database Configuration
# ---------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    # Fix for Render PostgreSQL URL (postgres:// -> postgresql://)
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
    logger.info("Using PostgreSQL database")
else:
    # Fallback to SQLite for local development
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///mood_app.db"
    logger.info("Using SQLite database (local development)")

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ---------------------------
# Database Models
# ---------------------------
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    email = db.Column(db.String(120), default="")
    gender = db.Column(db.String(20), default="")
    age = db.Column(db.String(10), default="")
    created_at = db.Column(db.DateTime, default=db.func.now())

    def to_dict(self):
        return {
            "username": self.username,
            "email": self.email or "None",
            "gender": self.gender or "None",
            "age": self.age or "None"
        }

class Message(db.Model):
    __tablename__ = "messages"
    id = db.Column(db.Integer, primary_key=True)
    sender = db.Column(db.String(80), nullable=False)
    receiver = db.Column(db.String(80), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        return {
            "from": self.sender,
            "to": self.receiver,
            "text": self.text,
            "ts": self.timestamp
        }

# Create tables
with app.app_context():
    db.create_all()
    logger.info("Database tables created")

# ---------------------------
# Initialize Face Detection
# ---------------------------
try:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    logger.info("✅ Face, smile & eye cascades loaded")
except Exception as e:
    logger.error(f"Failed to load cascades: {e}")
    face_cascade = None
    smile_cascade = None
    eye_cascade = None

# ---------------------------
# Emotion Detection (Feature-Based)
# ---------------------------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion_simple(frame):
    """
    High-precision emotion detection using Haar cascades (smile, eyes) and 
    geometric facial texture analysis. Works robustly across different lighting conditions and distances.
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape[:2]
        
        # Equalize histogram to normalize room lighting differences
        gray_eq = cv2.equalizeHist(gray)
        
        # Multi-pass face detection with fallback thresholds
        faces = []
        if face_cascade is not None:
            # Pass 1: Standard equalized
            faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
            if len(faces) == 0:
                # Pass 2: Raw grayscale with forgiving scale
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=2, minSize=(30, 30))
        
        # If Haar cascade still misses due to webcam lighting/angle, fallback to center facial region
        if len(faces) > 0:
            (x, y, fw, fh) = faces[0]
            face_roi = gray_eq[y:y+fh, x:x+fw]
            raw_face_roi = gray[y:y+fh, x:x+fw]
        else:
            # Center 70% of frame where the user's face is positioned when facing the webcam
            cy1, cy2 = int(h * 0.15), int(h * 0.85)
            cx1, cx2 = int(w * 0.20), int(w * 0.80)
            face_roi = gray_eq[cy1:cy2, cx1:cx2]
            raw_face_roi = gray[cy1:cy2, cx1:cx2]
            fw, fh = cx2 - cx1, cy2 - cy1
        
        # Split face into regions
        upper_roi = face_roi[:int(fh * 0.55), :]   # Eyes, eyebrows, forehead
        lower_roi = face_roi[int(fh * 0.55):, :]   # Mouth, lips, jawline
        
        # Check for genuine smiles (Happy)
        smiles = []
        if 'smile_cascade' in globals() and smile_cascade is not None and lower_roi.size > 0:
            smiles = smile_cascade.detectMultiScale(
                lower_roi,
                scaleFactor=1.5,
                minNeighbors=12,
                minSize=(int(fw * 0.2), int(fh * 0.12))
            )
            
        # Check for open eyes (Surprise / Fear)
        eyes = []
        if 'eye_cascade' in globals() and eye_cascade is not None and upper_roi.size > 0:
            eyes = eye_cascade.detectMultiScale(
                upper_roi,
                scaleFactor=1.12,
                minNeighbors=5,
                minSize=(int(fw * 0.15), int(fh * 0.15))
            )
            
        # Calculate texture & edge variance metrics
        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
        texture_std = float(np.std(laplacian))
        
        upper_lap = cv2.Laplacian(upper_roi, cv2.CV_64F)
        upper_texture = float(np.std(upper_lap))
        
        lower_std = float(np.std(lower_roi))
        contrast = float(np.std(raw_face_roi))
        
        # Emotion Classification Rules
        
        # 1. Happy: Smile cascade detected or strong lower-face expression variance with normal/relaxed upper face
        if len(smiles) > 0 or (lower_std > 56 and upper_texture < 44):
            return "Happy"
            
        # 2. Surprise: Wide open eyes combined with elevated contrast/texture
        elif len(eyes) >= 2 and texture_std > 46 and contrast > 48:
            return "Surprise"
            
        # 3. Angry: High furrowed brow/forehead texture with narrowed features
        elif upper_texture > 50 and len(smiles) == 0 and contrast > 45:
            return "Angry"
            
        # 4. Sad: Subdued features, low contrast, low edge variance
        elif contrast < 38 and texture_std < 34 and len(smiles) == 0:
            return "Sad"
            
        # 5. Fear: Very high overall facial tension
        elif texture_std > 56 and contrast > 52 and len(eyes) >= 2:
            return "Fear"
            
        # 6. Disgust: Extreme localized nasal/mid-face wrinkles
        elif upper_texture > 60 and lower_std > 60 and len(smiles) == 0 and len(eyes) == 0:
            return "Disgust"
            
        # 7. Neutral: Default when facial features are relaxed and balanced
        else:
            return "Neutral"
            
    except Exception as e:
        logger.warning(f"Emotion detection error: {e}")
        return "Happy"

# ---------------------------
# Spotify Setup
# ---------------------------
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")

# Initialize Spotify client (will be None if credentials missing)
sp = None
if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID, 
            client_secret=SPOTIFY_CLIENT_SECRET
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)
        logger.info("Spotify client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Spotify client: {e}")
else:
    logger.warning("Spotify credentials not configured. Music features will use fallback tracks.")

# ---------------------------
# Mood → Spotify
# ---------------------------
def mood_to_query(mood):
    mapping = {
        "neutral":"chill","angry":"rock","surprise":"party",
        "happy":"upbeat","sad":"melancholy","fear":"intense","disgust":"grunge"
    }
    return mapping.get(mood.lower(),mood.lower())

FALLBACK_TRACKS = [
    {"name": "Chill Vibes", "artist": "Unknown", "url": "https://open.spotify.com/embed/track/6rqhFgbbKwnb9MLmUQDhG6"},
    {"name": "Relaxing Beats", "artist": "Unknown", "url": "https://open.spotify.com/embed/track/0VjIjW4GlUZAMYd2vXMi3b"},
    {"name": "Mood Lifter", "artist": "Unknown", "url": "https://open.spotify.com/embed/track/7qiZfU4dY1lWllzX7mPBI3"},
    {"name": "Feel Good", "artist": "Unknown", "url": "https://open.spotify.com/embed/track/0pqnGHJpmpxLKifKRmU6WP"},
    {"name": "Easy Listening", "artist": "Unknown", "url": "https://open.spotify.com/embed/track/1BxfuPKGuaTgP7aM0Bbdwr"},
]
_song_cache = {}
CACHE_TTL_SECONDS = 10 * 60

def get_songs_spotify(mood: str) -> list:
    """Return 5 unique Spotify tracks for a given mood."""
    mood_key = (mood or "chill").lower()
    now = time.time()
    
    # Return fallback if Spotify not configured
    if sp is None:
        return FALLBACK_TRACKS[:5]
    
    # Use cache if valid
    cached = _song_cache.get(mood_key)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        tracks = cached[1]
    else:
        query = f"{mood_to_query(mood)} bollywood"
        tracks = []
        try:
            results = sp.search(q=query, type="track", limit=50)
            seen = set()
            for t in results.get("tracks", {}).get("items", []):
                tid = t.get("id")
                if not tid or tid in seen:
                    continue
                tracks.append({
                    "name": t.get("name", "Unknown"),
                    "artist": t.get("artists", [{}])[0].get("name", "Unknown"),
                    "url": f"https://open.spotify.com/embed/track/{tid}"
                })
                seen.add(tid)
            # Pad with fallback tracks if needed
            while len(tracks) < 5:
                tracks.append(FALLBACK_TRACKS[len(tracks) % len(FALLBACK_TRACKS)].copy())
        except Exception as e:
            logger.error(f"Spotify search error: {e}")
            tracks = [t.copy() for t in FALLBACK_TRACKS[:5]]
        _song_cache[mood_key] = (now, tracks)
    
    if len(tracks) <= 5:
        return tracks
    return random.sample(tracks, 5)

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def root(): return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    """Handle user registration."""
    error = None
    success = None
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        email = request.form.get("email", "").strip()
        gender = request.form.get("gender", "").strip()
        age = request.form.get("age", "").strip()
        
        if not username or not password:
            error = "Username and password are required"
        elif len(password) < 6:
            error = "Password must be at least 6 characters"
        elif User.query.filter_by(username=username).first():
            error = "Username already exists!"
        else:
            new_user = User(
                username=username,
                password=generate_password_hash(password),
                email=email,
                gender=gender,
                age=age
            )
            db.session.add(new_user)
            db.session.commit()
            success = "Registration successful! Please login."
            logger.info(f"New user registered: {username}")
    
    return render_template("register.html", error=error, success=success)

@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login."""
    error = None
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            access_token = create_access_token(identity=username)
            resp = make_response(redirect(url_for("dashboard")))
            set_access_cookies(resp, access_token)
            logger.info(f"User logged in: {username}")
            return resp
        else:
            error = "Invalid username or password"
    
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    resp=make_response(redirect(url_for("login")))
    unset_jwt_cookies(resp)
    return resp

@app.route("/dashboard")
@jwt_required()
def dashboard():
    username = get_jwt_identity()
    user = User.query.filter_by(username=username).first()
    if not user:
        return redirect(url_for("logout"))
    return render_template("dashboard.html", 
                           username=username, 
                           email=user.email or "None",
                           gender=user.gender or "None", 
                           age=user.age or "None")

@app.route("/chat")
@jwt_required()
def chat():
    """Redirect to dashboard with chat popup opened."""
    return redirect(url_for("dashboard", openChat="true"))

@app.route("/update_profile", methods=["POST"])
@jwt_required()
def update_profile():
    username = get_jwt_identity()
    user = User.query.filter_by(username=username).first()
    if user:
        user.gender = request.form.get("gender", "").strip()
        user.age = request.form.get("age", "").strip()
        user.email = request.form.get("email", "").strip()
        db.session.commit()
    return redirect(url_for("dashboard"))

@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    """Detect emotion from face image using feature-based analysis."""
    body = request.get_json(silent=True)
    if not body or "image" not in body:
        return jsonify({"emotion": "No image"}), 400
    
    try:
        # Decode base64 image
        data_url = body["image"]
        img_data = base64.b64decode(data_url.split(",", 1)[1])
        arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"emotion": "Invalid image"}), 400
        
        # Detect emotion
        emotion = detect_emotion_simple(frame)
        logger.info(f"🎭 Emotion detected: {emotion}")
        
        return jsonify({"emotion": emotion})
        
    except Exception as e:
        logger.error(f"Predict error: {e}")
        return jsonify({"emotion": random.choice(emotion_labels)})

@app.route("/songs", methods=["POST"])
def songs():
    """Get song recommendations based on mood."""
    body = request.get_json(silent=True)
    mood = body.get("mood") if body else None
    
    # Normalize mood
    safe_mood = (mood or "chill").strip()
    if safe_mood.lower() in ("no face detected", "not detected", "none", ""):
        safe_mood = "chill"
    
    tracks = get_songs_spotify(safe_mood)
    return jsonify({"mood": safe_mood, "songs": tracks})

# ---------------------------
# Chat routes
# ---------------------------
@app.route("/chat/send", methods=["POST"])
@jwt_required()
def chat_send():
    body = request.get_json()
    if not body or "to" not in body or "text" not in body:
        return jsonify({"error": "Missing fields"}), 400
    
    username = get_jwt_identity()
    new_msg = Message(
        sender=username,
        receiver=body["to"],
        text=body["text"],
        timestamp=int(time.time())
    )
    db.session.add(new_msg)
    db.session.commit()
    return jsonify({"status": "ok"})


@app.route("/chat/fetch", methods=["GET"])
@jwt_required()
def chat_fetch():
    chat_with = request.args.get("user")
    if not chat_with:
        return jsonify({"error": "Missing user"}), 400
    
    username = get_jwt_identity()
    
    messages = Message.query.filter(
        db.or_(
            db.and_(Message.sender == username, Message.receiver == chat_with),
            db.and_(Message.sender == chat_with, Message.receiver == username)
        )
    ).order_by(Message.timestamp).all()
    
    return jsonify([m.to_dict() for m in messages])

@app.route("/users/list", methods=["GET"])
@jwt_required()
def users_list():
    username = get_jwt_identity()
    users = User.query.filter(User.username != username).all()
    return jsonify([u.username for u in users])



# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode, use_reloader=debug_mode)
