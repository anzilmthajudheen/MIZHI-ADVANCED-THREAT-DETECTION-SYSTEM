import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time

class MIZHI_WeaponDetector:
    def __init__(self):
        self.yolo_model = YOLO("yolov8m.pt")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Define weapon classes and their categories
        self.weapon_classes = {
            # Real weapons (RED BOX)
            'knife': 'real_weapon',
            'gun': 'real_weapon',
            'pistol': 'real_weapon',
            'rifle': 'real_weapon',
            'sword': 'real_weapon',
            'axe': 'real_weapon',
            'hammer': 'real_weapon',
            'baseball bat': 'real_weapon',
            'scissors': 'real_weapon',
            
            # Devices that might show fake weapons (YELLOW BOX)
            'cell phone': 'potential_fake',
            'laptop': 'potential_fake',
            'tv': 'potential_fake',
            'monitor': 'potential_fake',
            'tablet': 'potential_fake'
        }
        
        self.feature_extractor = None
        self.lstm_model = None
        self.sequence = []
        self.sequence_length = 30
        
        # GUI variables
        self.root = None
        self.video_source = 0
        self.cap = None
        self.is_running = False
        self.video_label = None
        
    def create_feature_extractor(self):
        """Create ResNet50-based feature extractor"""
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet", 
            include_top=False, 
            input_shape=(128, 128, 3)
        )
        model = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))
        return model
    
    def create_lstm_model(self, input_shape):
        """Create LSTM model for behavioral analysis"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(32, return_sequences=True),
            Dropout(0.3),
            LSTM(16),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    def load_trained_models(self):
        """Load your trained models"""
        try:
            # Load CNN model (optional)
            self.cnn_model = tf.keras.models.load_model('models/mizhi_cnn_final.h5')

            # Load LSTM model
            self.lstm_model = tf.keras.models.load_model('models/mizhi_lstm_final.h5')

            # Load feature extractor
            self.feature_extractor = tf.keras.models.load_model('models/mizhi_feature_extractor.h5')

            print("✅ All trained models loaded successfully!")
            self.status_var.set("Models loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.status_var.set(f"Error loading models: {e}")
            return False
    def detect_objects(self, frame):
        """Enhanced object detection with weapon classification"""
        results = self.yolo_model(frame, conf=0.3)
        detected_weapons = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = self.yolo_model.names[class_id]
                
                # Check if detected object is a weapon or potential fake weapon source
                weapon_type = self.classify_detection(label, confidence)
                
                if weapon_type:
                    detected_weapons.append({
                        'label': label,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'type': weapon_type
                    })
                    
                    # Draw bounding box with appropriate color
                    color = self.get_box_color(weapon_type)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add text with background
                    text = f"{label}: {confidence:.2f}"
                    self.draw_text_with_background(frame, text, (x1, y1 - 10), color)
        
        return frame, detected_weapons
    
    def classify_detection(self, label, confidence):
        """Classify detection as real weapon, fake weapon, or suspicious"""
        label_lower = label.lower()
        
        # Check for real weapons
        for weapon, category in self.weapon_classes.items():
            if weapon in label_lower:
                if category == 'real_weapon' and confidence > 0.4:
                    return 'real_weapon'
                elif category == 'potential_fake' and confidence > 0.5:
                    return 'potential_fake'
        
        # Check for other suspicious objects
        suspicious_objects = ['bottle', 'stick', 'tool', 'wrench', 'screwdriver']
        for obj in suspicious_objects:
            if obj in label_lower and confidence > 0.6:
                return 'suspicious'
        
        return None
    
    def get_box_color(self, weapon_type):
        """Get color for bounding box based on weapon type"""
        colors = {
            'real_weapon': (0, 0, 255),      # Red
            'potential_fake': (0, 255, 255), # Yellow
            'suspicious': (0, 165, 255)       # Orange
        }
        return colors.get(weapon_type, (255, 255, 255))
    
    def draw_text_with_background(self, frame, text, position, color):
        """Draw text with background for better visibility"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (position[0], position[1] - text_height - baseline),
                     (position[0] + text_width, position[1] + baseline),
                     color, -1)
        
        # Draw text
        cv2.putText(frame, text, position, font, font_scale, (255, 255, 255), thickness)
    
    def detect_pose_and_behavior(self, frame):
        """Detect pose and analyze for suspicious behavior"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        suspicious_behavior = False
        
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Analyze pose for suspicious behavior
            landmarks = results.pose_landmarks.landmark
            
            # Check for aggressive postures
            if self.is_aggressive_posture(landmarks):
                suspicious_behavior = True
        
        return frame, suspicious_behavior
    
    def is_aggressive_posture(self, landmarks):
        """Analyze pose landmarks for aggressive behavior"""
        try:
            # Get key points
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Check if arms are raised (potential threatening gesture)
            if (left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y):
                return True
            
            # Add more pose analysis rules here
            
        except:
            pass
        
        return False
    
    def analyze_sequence(self, frame):
        """Analyze frame sequence for behavioral patterns"""
        if self.feature_extractor is None:
            return 0.0
        
        frame_resized = cv2.resize(frame, (128, 128)) / 255.0
        features = self.feature_extractor.predict(frame_resized[np.newaxis, ...], verbose=0)[0]
        self.sequence.append(features)
        
        if len(self.sequence) > self.sequence_length:
            self.sequence.pop(0)
        
        if len(self.sequence) == self.sequence_length and self.lstm_model is not None:
            sequence_array = np.array(self.sequence)[np.newaxis, ...]
            prediction = self.lstm_model.predict(sequence_array, verbose=0)[0][0]
            return prediction
        
        return 0.0
    
    def add_alert_overlay(self, frame, detected_weapons, suspicious_behavior, behavior_score):
        """Add alert overlays to the frame"""
        height, width = frame.shape[:2]
        
        # Create alert panel
        alert_panel = np.zeros((100, width, 3), dtype=np.uint8)
        
        # Check for alerts
        alerts = []
        
        # Real weapon alerts
        real_weapons = [w for w in detected_weapons if w['type'] == 'real_weapon']
        if real_weapons:
            alerts.append(f"⚠️ REAL WEAPON DETECTED: {', '.join([w['label'] for w in real_weapons])}")
        
        # Fake weapon alerts
        fake_weapons = [w for w in detected_weapons if w['type'] == 'potential_fake']
        if fake_weapons:
            alerts.append(f"📱 POSSIBLE FAKE WEAPON: {', '.join([w['label'] for w in fake_weapons])}")
        
        # Suspicious activity alerts
        suspicious_objects = [w for w in detected_weapons if w['type'] == 'suspicious']
        if suspicious_objects:
            alerts.append(f"🔍 SUSPICIOUS OBJECT: {', '.join([w['label'] for w in suspicious_objects])}")
        
        if suspicious_behavior or behavior_score > 0.7:
            alerts.append("⚡ SUSPICIOUS BEHAVIOR DETECTED")
        
        # Draw alerts
        y_offset = 20
        for alert in alerts:
            cv2.putText(alert_panel, alert, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(alert_panel, f"MIZHI System - {timestamp}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Combine frame with alert panel
        combined_frame = np.vstack([alert_panel, frame])
        
        return combined_frame
    
    def create_gui(self):
        """Create the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("MIZHI - Advanced Weapon Detection System")
        self.root.geometry("1200x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="MIZHI - Advanced Weapon Detection System", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Camera selection
        ttk.Label(control_frame, text="Camera:").grid(row=0, column=0, padx=5)
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(control_frame, textvariable=self.camera_var, 
                                   values=["0", "1", "2", "3"], width=10)
        camera_combo.grid(row=0, column=1, padx=5)
        
        # Video file selection
        ttk.Button(control_frame, text="Select Video File", 
                  command=self.select_video_file).grid(row=0, column=2, padx=5)
        
        # Control buttons
        ttk.Button(control_frame, text="Start Detection", 
                  command=self.start_detection).grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="Stop Detection", 
                  command=self.stop_detection).grid(row=0, column=4, padx=5)
        ttk.Button(control_frame, text="Train Model", 
                  command=self.train_model).grid(row=0, column=5, padx=5)
        
        # Video display
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Legend
        legend_frame = ttk.LabelFrame(main_frame, text="Detection Legend", padding="10")
        legend_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(legend_frame, text="🔴 Red Box: Real Weapons", 
                 foreground="red").grid(row=0, column=0, padx=10)
        ttk.Label(legend_frame, text="🟡 Yellow Box: Possible Fake Weapons", 
                 foreground="orange").grid(row=0, column=1, padx=10)
        ttk.Label(legend_frame, text="🟠 Orange Box: Suspicious Objects", 
                 foreground="#FF8C00").grid(row=0, column=2, padx=10)
    
    def select_video_file(self):
        """Select video file for processing"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.video_source = file_path
            self.status_var.set(f"Video file selected: {os.path.basename(file_path)}")
    
    def start_detection(self):
        """Start the detection process"""
        if not self.is_running:
            self.is_running = True
            self.status_var.set("Starting detection...")
            
            # Initialize models if not already done
            if self.feature_extractor is None:
                self.feature_extractor = self.create_feature_extractor()
            
            # Start detection in a separate thread
            detection_thread = threading.Thread(target=self.run_detection)
            detection_thread.daemon = True
            detection_thread.start()
    
    def stop_detection(self):
        """Stop the detection process"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.status_var.set("Detection stopped")
    
    def train_model(self):
        """Train the LSTM model"""
        dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        if dataset_path:
            self.status_var.set("Training model...")
            # Run training in a separate thread
            training_thread = threading.Thread(target=self.train_lstm_model, args=(dataset_path,))
            training_thread.daemon = True
            training_thread.start()
    
    def run_detection(self):
        """Main detection loop"""
        try:
            # Initialize video capture
            if isinstance(self.video_source, str) and not self.video_source.isdigit():
                self.cap = cv2.VideoCapture(self.video_source)
            else:
                self.cap = cv2.VideoCapture(int(self.camera_var.get()))
            
            if not self.cap.isOpened():
                self.status_var.set("Error: Could not open video source")
                return
            
            self.status_var.set("Detection running...")
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Detect objects
                frame, detected_weapons = self.detect_objects(frame)
                
                # Detect pose and behavior
                frame, suspicious_behavior = self.detect_pose_and_behavior(frame)
                
                # Analyze sequence
                behavior_score = self.analyze_sequence(frame)
                
                # Add alert overlay
                frame = self.add_alert_overlay(frame, detected_weapons, 
                                             suspicious_behavior, behavior_score)
                
                # Update GUI
                self.update_video_display(frame)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.03)
        
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            self.is_running = False
    
    def update_video_display(self, frame):
        """Update the video display in the GUI"""
        if self.video_label:
            # Resize frame for display
            display_frame = cv2.resize(frame, (800, 600))
            
            # Convert to RGB for Tkinter
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk
    
    def train_lstm_model(self, dataset_path):
        """Train the LSTM model with the provided dataset"""
        try:
            # Load and preprocess data
            frames, labels = self.load_frame_dataset(dataset_path, sample_limit=5000)
            
            if len(frames) == 0:
                self.status_var.set("Error: No valid frames found in dataset")
                return
            
            # Create feature extractor if not exists
            if self.feature_extractor is None:
                self.feature_extractor = self.create_feature_extractor()
            
            # Extract features
            features = []
            for frame in frames:
                feature = self.feature_extractor.predict(frame[np.newaxis, ...], verbose=0)[0]
                features.append(feature)
            
            features = np.array(features)
            
            # Create sequences
            sequences, seq_labels = [], []
            for i in range(len(features) - self.sequence_length):
                sequences.append(features[i:i+self.sequence_length])
                seq_labels.append(labels[i+self.sequence_length])
            
            sequences = np.array(sequences)
            seq_labels = np.array(seq_labels)
            
            # Create and train LSTM model
            self.lstm_model = self.create_lstm_model((self.sequence_length, features.shape[1]))
            
            # Train model
            self.lstm_model.fit(sequences, seq_labels, epochs=10, batch_size=8, 
                              validation_split=0.2, verbose=1)
            
            self.status_var.set("Model training completed successfully")
            
        except Exception as e:
            self.status_var.set(f"Training error: {str(e)}")
    
    def load_frame_dataset(self, dataset_path, sample_limit=None):
        """Load frame dataset for training"""
        frames, labels = [], []
        class_dirs = {'train': 1, 'valid': 0}
        
        for class_name, label in class_dirs.items():
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.exists(class_path):
                continue
            
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image = cv2.imread(file_path)
                    if image is None:
                        continue
                    image = cv2.resize(image, (128, 128)) / 255.0
                    frames.append(image)
                    labels.append(label)
                if sample_limit and len(frames) >= sample_limit:
                    break
        
        return np.array(frames), np.array(labels)
    
    def run(self):
        """Run the application"""
        self.create_gui()
        self.load_trained_models()  # <--- Auto-load models here
        self.root.mainloop()


# Main execution
if __name__ == "__main__":
    app = MIZHI_WeaponDetector()
    app.run()