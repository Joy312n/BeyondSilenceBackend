import cv2
import numpy as np
import mediapipe as mp
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import eventlet

# --- MODIFIED: Replaced Event with a simple boolean flag ---
camera_is_active = False

# --- Initialize Flask and SocketIO ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key' # Change this
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Load Model and Classes ---
print("Loading model and class data...")
model = load_model("best_model.h5")
classes = np.load("classes.npy", allow_pickle=True) 
threshold = 0.8 # Confidence threshold

# Get the input shape the model expects
try:
    input_shape = model.layers[0].input_shape[0] 
    MAX_LEN = input_shape[1]
    FEATURES = input_shape[2]
    print(f"Model loaded. Expects input shape: (None, {MAX_LEN}, {FEATURES})")
except Exception as e:
    print(f"Error inspecting model input shape: {e}")
    print("Using fallback MAX_LEN=30. This might be wrong.")
    MAX_LEN = 30 # Fallback

# --- Mediapipe Setup ---
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Drawing styles
pose_style = mp_drawing.DrawingSpec(thickness=2, circle_radius=3, color=(0, 255, 0))
hand_style = mp_drawing.DrawingSpec(thickness=3, circle_radius=2, color=(255, 0, 0))

def mediapipe_detection(image, holistic, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results_holistic = holistic.process(image_rgb)
    results_hands = hands.process(image_rgb)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results_hands.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            if handedness.classification[0].label == 'Left':
                results_holistic.left_hand_landmarks = hand_landmarks
            elif handedness.classification[0].label == 'Right':
                results_holistic.right_hand_landmarks = hand_landmarks
    return image, results_holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                      for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face_indices = [1, 33, 263, 133, 362]
    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[i].x,
                          results.face_landmarks.landmark[i].y,
                          results.face_landmarks.landmark[i].z]
                         for i in face_indices]).flatten()
    else:
        face = np.zeros(len(face_indices) * 3)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def draw_custom_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, pose_style, pose_style)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_style, hand_style)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_style, hand_style)
    if results.face_landmarks:
        face_indices = [1, 33, 263, 133, 362]
        h, w, _ = image.shape
        for idx in face_indices:
            landmark = results.face_landmarks.landmark[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 2, (0, 0, 255), cv2.FILLED)

# --- Global variable for the camera thread ---
camera_thread = None

def video_processing_thread():
    """This function runs in a background thread to process video."""
    # --- FIX: Declare global at the TOP of the function scope ---
    global camera_is_active 
    
    print("Starting video processing thread...")
    cap = cv2.VideoCapture(0)
    sequence = []
    current_prediction_text = "Unknown"

    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic, \
         mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=2) as hands:
        
        while True:
            if camera_is_active:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame. Pausing.")
                    camera_is_active = False # We can now set it directly
                    continue
                
                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, holistic, hands)
                draw_custom_landmarks(image, results)
                
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-MAX_LEN:]
                
                if len(sequence) > 10: # Start predicting
                    input_data = pad_sequences([sequence], padding='post', dtype='float32', maxlen=MAX_LEN)
                    prediction = model.predict(input_data, verbose=0)[0]
                    pred_class = np.argmax(prediction)
                    confidence = prediction[pred_class]

                    if confidence > threshold:
                        current_prediction_text = f"{classes[pred_class]} ({confidence:.2f})"
                    else:
                        current_prediction_text = "Unknown"
                else:
                    current_prediction_text = "Collecting..."

                cv2.rectangle(image, (0,0), (640, 50), (0,0,0), cv2.FILLED)
                cv2.putText(image, current_prediction_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)
                
                _, buffer = cv2.imencode('.jpg', image)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                socketio.emit('video_feed', {
                    'image': 'data:image/jpeg;base64,' + frame_data,
                    'prediction': current_prediction_text
                })
            
            socketio.sleep(0.03)

    cap.release()
    print("Video processing thread stopped.")

@socketio.on('connect')
def handle_connect(auth=None): 
    global camera_thread
    global camera_is_active 
    print('Client connected')
    
    if camera_thread is None:
        camera_thread = socketio.start_background_task(target=video_processing_thread)
    
    camera_is_active = False 

@socketio.on('disconnect')
def handle_disconnect():
    global camera_is_active 
    print('Client disconnected')
    camera_is_active = False 

@socketio.on('start_camera')
def handle_start_camera():
    global camera_is_active 
    print("Received start_camera event")
    camera_is_active = True 

@socketio.on('stop_camera')
def handle_stop_camera():
    global camera_is_active 
    print("Received stop_camera event")
    camera_is_active = False 

if __name__ == '__main__':
    print("Starting Flask-SocketIO server at http://127.0.0.1:5000")
    socketio.run(app, host='127.0.0.1', port=5000, debug=True)