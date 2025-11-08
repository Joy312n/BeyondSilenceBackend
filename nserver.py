# servernew.py
import base64
import numpy as np
from flask import Flask
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Global flags / state (same behaviour as your original) ---
static_sequence_str = ""
static_last_pred = ""
static_cooldown_frames = 15
static_counter = 0
no_hand_counter = 0
no_hand_threshold = 30

# --- Flask/SocketIO ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Load dynamic model ---
print("Loading DYNAMIC model and class data...")
try:
    dynamic_model = load_model("best_model_old.h5")
    dynamic_classes = np.load("classes.npy", allow_pickle=True)
    dynamic_threshold = 0.8
    model_input_shape = dynamic_model.input_shape
    MAX_LEN = model_input_shape[1]
    FEATURES = model_input_shape[2]
    print(f"Dynamic model loaded. Expects shape: (None, {MAX_LEN}, {FEATURES})")
except Exception as e:
    print(f"Error loading DYNAMIC model: {e}")
    MAX_LEN = 30
    FEATURES = None

# --- Load static model & classes (same as original) ---
print("Loading STATIC model and class data...")
try:
    static_model = load_model("asl_static_model1.h5")
    static_class_names = [
        'A','B','C','D','E','F','G','H','I','J',
        'K','L','M','N','O','P','Q','R','S','T',
        'U','V','W','X','Y','Z'
    ]
    static_threshold = 0.8
    # static input len if available
    try:
        static_input_len = static_model.input_shape[1]
    except Exception:
        static_input_len = None
    print("Static model loaded.")
except Exception as e:
    print(f"Error loading STATIC model (asl_static_model1.h5): {e}")
    static_model = None
    static_input_len = None

# --- Per-client dynamic sequence buffer map (keep sequences per sid) ---
from collections import defaultdict, deque
client_dynamic_sequences = defaultdict(lambda: deque(maxlen=MAX_LEN))

# --- Helper functions for server-side prediction (mirrors previous logic) ---

def predict_dynamic_for_client(sid):
    """Take stored dynamic sequence for client sid, pad and predict (if enough frames)."""
    seq = list(client_dynamic_sequences[sid])
    if len(seq) == 0:
        return "Collecting...", None
    if len(seq) < 11:
        return "Collecting...", None

    input_data = pad_sequences([seq], padding='post', dtype='float32', maxlen=MAX_LEN)
    prediction = dynamic_model.predict(input_data, verbose=0)[0]
    pred_class = int(np.argmax(prediction))
    confidence = float(prediction[pred_class])

    if confidence > dynamic_threshold:
        label = f"{dynamic_classes[pred_class]} ({confidence:.2f})"
    else:
        label = "Unknown"
    return label, confidence

def predict_static_from_coords(coords_flat):
    """coords_flat: flattened (21*3,) values already normalized on client or raw coords.
       We expect the client to send the same coords_norm produced by the frontend normalize function.
    """
    global static_sequence_str, static_last_pred, static_counter

    if static_model is None:
        return "Static model error", None

    # Coerce coords into shape (1, N)
    coords_arr = np.array(coords_flat, dtype='float32').reshape(1, -1)
    prediction = static_model.predict(coords_arr, verbose=0)
    pred_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    predicted_label = static_class_names[pred_idx]

    current_prediction_text = f"{predicted_label} ({confidence:.2f})"

    # Static sequence logic (identical behavior to original)
    if confidence > static_threshold:
        if predicted_label != static_last_pred and static_counter == 0:
            static_sequence_str += predicted_label
            static_last_pred = predicted_label
            static_counter = static_cooldown_frames
        elif static_counter > 0:
            static_counter = max(0, static_counter - 1)

    # decrement counter when same label (preserve original logic)
    if predicted_label == static_last_pred and static_counter > 0:
        static_counter = max(0, static_counter - 1)

    return current_prediction_text, confidence

# --- Socket handlers ---

@socketio.on('connect')
def handle_connect(auth=None):
    sid = getattr(socketio, 'sid', None)
    print(f"Client connected: {request_sid()}")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request_sid()
    print(f"Client disconnected: {sid}")
    # cleanup per-client data
    if sid in client_dynamic_sequences:
        del client_dynamic_sequences[sid]

def request_sid():
    # helper to get sid in different Flask-SocketIO contexts
    try:
        from flask import request
        return request.sid
    except Exception:
        return None

@socketio.on('request_model_info')
def handle_request_model_info():
    """Frontend can request max_len/features so it builds sequences to correct shape."""
    emit('model_info', {
        'max_len': MAX_LEN,
        'features': FEATURES,
        'static_input_len': static_input_len
    })

@socketio.on('frame_keypoints')
def handle_frame_keypoints(data):
    """
    Expects:
      data = {
        'mode': 'dynamic' or 'static',
        'keypoints': <list/flat array>
      }
    For dynamic: keypoints should be the full flattened keypoint vector for a single frame,
                 matching extract_keypoints_dynamic output in your original code.
    For static: keypoints should be the normalized flattened hand coords (21*3) matching normalize_landmarks_static output (flattened).
    """
    sid = request_sid()
    mode = data.get('mode', 'dynamic')
    k = data.get('keypoints', None)

    if k is None:
        emit('video_feed', {'prediction': 'Error - no keypoints', 'mode': mode})
        return

    if mode == 'dynamic':
        # push into client's deque
        client_dynamic_sequences[sid].append(k.tolist() if isinstance(k, np.ndarray) else k)
        label, conf = predict_dynamic_for_client(sid)
        # emit same structure as original 'video_feed' for minimal frontend changes:
        emit('video_feed', {
            'image': None,  # no frame image from server anymore
            'prediction': label,
            'mode': 'dynamic'
        })
    elif mode == 'static':
        # k should be normalized coords flattened
        label, conf = predict_static_from_coords(k)
        # also include formed sequence string
        emit('video_feed', {
            'image': None,
            'prediction': label,
            'sequence': static_sequence_str,
            'mode': 'static'
        })
    else:
        emit('video_feed', {'image': None, 'prediction': 'Invalid mode', 'mode': mode})

if __name__ == '__main__':
    print("Starting Flask-SocketIO server for keypoint-only predictions.")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
