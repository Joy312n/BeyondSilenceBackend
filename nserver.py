import os  # <-- Added for Render port
import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict, deque

# --- Flask/SocketIO ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Global Config (Loaded once) ---
dynamic_model = None
dynamic_classes = []
MAX_LEN = 30  # Default fallback
FEATURES = None # Default fallback
dynamic_threshold = 0.8

static_model = None
static_class_names = []
static_input_len = 63 # Default fallback
static_threshold = 0.8
static_cooldown_frames = 15
no_hand_threshold = 30

# --- Load dynamic model ---
print("Loading DYNAMIC model and class data...")
try:
    dynamic_model = load_model("best_model_old.h5")
    dynamic_classes = np.load("classes.npy", allow_pickle=True)
    model_input_shape = dynamic_model.input_shape
    MAX_LEN = model_input_shape[1]
    FEATURES = model_input_shape[2]
    print(f"Dynamic model loaded. Expects shape: (None, {MAX_LEN}, {FEATURES})")
except Exception as e:
    print(f"Error loading DYNAMIC model: {e}")
    print(f"Using fallback MAX_LEN={MAX_LEN}. FEATURES will be requested from client info.")

# --- Load static model ---
print("Loading STATIC model and class data...")
try:
    static_model = load_model("asl_static_model1.h5")
    static_class_names = [
        'A','B','C','D','E','F','G','H','I','J',
        'K','L','M','N','O','P','Q','R','S','T',
        'U','V','W','X','Y','Z'
    ]
    static_input_len = static_model.input_shape[1]
    print(f"Static model loaded. Expects input length: {static_input_len}")
except Exception as e:
    print(f"Error loading STATIC model (asl_static_model1.h5): {e}")
    static_model = None

# --- Per-client State Management ---

# Stores dynamic sequence data for each client (by session ID)
client_dynamic_sequences = defaultdict(lambda: deque(maxlen=MAX_LEN))

# **FIXED**: Stores static state for each client (by session ID)
def new_static_state():
    """Helper to create a new, blank state for a static user."""
    return {
        "sequence": "",
        "last_pred": "",
        "counter": 0,
        "no_hand": 0
    }
client_static_state = defaultdict(new_static_state)

# --- Prediction Helper Functions ---

def predict_dynamic_for_client(sid):
    """Pads and predicts the sequence for a specific client."""
    seq = list(client_dynamic_sequences[sid])
    if not seq or len(seq) < 11: # Start predicting after 10 frames
        return "Collecting...", None

    try:
        input_data = pad_sequences([seq], padding='post', dtype='float32', maxlen=MAX_LEN)
        prediction = dynamic_model.predict(input_data, verbose=0)[0]
        pred_class = int(np.argmax(prediction))
        confidence = float(prediction[pred_class])

        if confidence > dynamic_threshold:
            label = f"{dynamic_classes[pred_class]} ({confidence:.2f})"
        else:
            label = "Unknown"
        return label, confidence
    except Exception as e:
        print(f"Error during dynamic prediction: {e}")
        return "Prediction Error", None

def predict_static_from_coords(sid, coords_flat):
    """Predicts static sign and manages sequence string for a specific client."""
    if static_model is None:
        return "Static model error", None, ""

    # Get this client's specific state
    state = client_static_state[sid]
    
    try:
        coords_arr = np.array(coords_flat, dtype='float32').reshape(1, -1)
        
        # Check if hand is detected (any non-zero value)
        if np.any(coords_arr):
            state["no_hand"] = 0 # Reset no-hand counter
            
            prediction = static_model.predict(coords_arr, verbose=0)
            pred_idx = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            predicted_label = static_class_names[pred_idx]
            
            current_prediction_text = f"{predicted_label} ({confidence:.2f})"

            # Static sequence logic (using client's state)
            if confidence > static_threshold:
                if predicted_label != state["last_pred"] and state["counter"] == 0:
                    state["sequence"] += predicted_label
                    state["last_pred"] = predicted_label
                    state["counter"] = static_cooldown_frames
                elif state["counter"] > 0:
                    state["counter"] = max(0, state["counter"] - 1)
            
            if predicted_label == state["last_pred"] and state["counter"] > 0:
                state["counter"] = max(0, state["counter"] - 1)
        
        else:
            # No hand detected (all zeros)
            current_prediction_text = "..."
            confidence = None
            state["no_hand"] += 1
            if state["no_hand"] > no_hand_threshold and state["sequence"] and not state["sequence"].endswith(' '):
                state["sequence"] += " "
                state["no_hand"] = 0 # Reset after adding space

        return current_prediction_text, confidence, state["sequence"]

    except Exception as e:
        print(f"Error during static prediction: {e}")
        return "Prediction Error", None, state["sequence"]


# --- Socket Handlers ---

@socketio.on('connect')
def handle_connect(auth=None):
    sid = request.sid
    print(f"Client connected: {sid}")
    # Send model info as soon as client connects
    emit('model_info', {
        'max_len': MAX_LEN,
        'features': FEATURES,
        'static_input_len': static_input_len
    })

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"Client disconnected: {sid}")
    # Clean up data for the disconnected client
    if sid in client_dynamic_sequences:
        del client_dynamic_sequences[sid]
    if sid in client_static_state:
        del client_static_state[sid]

@socketio.on('request_model_info')
def handle_request_model_info():
    """Send model shape info to client on request."""
    emit('model_info', {
        'max_len': MAX_LEN,
        'features': FEATURES,
        'static_input_len': static_input_len
    })

@socketio.on('frame_keypoints')
def handle_frame_keypoints(data):
    """Main endpoint for receiving keypoints and sending back predictions."""
    sid = request.sid
    mode = data.get('mode', 'dynamic')
    k = data.get('keypoints', None)

    if k is None:
        return

    if mode == 'dynamic':
        if FEATURES is not None and len(k) != FEATURES:
            print(f"Warning: Client {sid} sent wrong dynamic features. Got {len(k)}, expected {FEATURES}")
            return
            
        client_dynamic_sequences[sid].append(k)
        label, conf = predict_dynamic_for_client(sid)
        
        emit('video_feed', {
            'image': None,
            'prediction': label,
            'sequence': '', # Dynamic mode doesn't have a sequence string
            'mode': 'dynamic'
        })
        
    elif mode == 'static':
        if static_input_len is not None and len(k) != static_input_len:
            print(f"Warning: Client {sid} sent wrong static features. Got {len(k)}, expected {static_input_len}")
            return

        label, conf, sequence_str = predict_static_from_coords(sid, k)
        
        emit('video_feed', {
            'image': None,
            'prediction': label,
            'sequence': sequence_str, # Send the client's current word
            'mode': 'static'
        })

if __name__ == '__main__':
    # **FIXED for Render**: Read port from environment, bind to 0.0.0.0
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask-SocketIO server on 0.0.0.0:{port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)