import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc

class MIZHITrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_size = (128, 128)
        self.sequence_length = 30
        self.feature_size = 2048  # ResNet50 feature size after global average pooling

    def load_images(self, max_images=None):
        """Load half images from train/valid folders"""
        print("Loading images from dataset...")

        images, labels = [], []
        train_loaded, valid_loaded = 0, 0

        max_per_split = (max_images // 2) if max_images else None

        # Load train images (label=1)
        train_files = []
        train_path = os.path.join(self.dataset_path, 'train')
        if os.path.exists(train_path):
            train_files = [f for f in os.listdir(train_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if max_per_split:
                train_files = train_files[:max_per_split]

        for i, filename in enumerate(train_files):
            filepath = os.path.join(train_path, filename)
            img = cv2.imread(filepath)
            if img is not None:
                img = cv2.resize(img, self.img_size).astype(np.float32) / 255.0
                images.append(img)
                labels.append(1)
                train_loaded += 1
                if train_loaded % 100 == 0:
                    print(f"Loaded {train_loaded} train images...")

        # Load valid images (label=0)
        valid_files = []
        valid_path = os.path.join(self.dataset_path, 'valid')
        if os.path.exists(valid_path):
            valid_files = [f for f in os.listdir(valid_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if max_per_split:
                valid_files = valid_files[:max_per_split]

        for i, filename in enumerate(valid_files):
            filepath = os.path.join(valid_path, filename)
            img = cv2.imread(filepath)
            if img is not None:
                img = cv2.resize(img, self.img_size).astype(np.float32) / 255.0
                images.append(img)
                labels.append(0)
                valid_loaded += 1
                if valid_loaded % 100 == 0:
                    print(f"Loaded {valid_loaded} valid images...")

        print(f"Total images loaded: {len(images)} (train: {train_loaded}, valid: {valid_loaded})")
        return np.array(images), np.array(labels)

    def create_cnn_model(self):
        """CNN architecture"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            BatchNormalization(), MaxPooling2D(2,2),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(), MaxPooling2D(2,2),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(), MaxPooling2D(2,2),
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(), MaxPooling2D(2,2),
            Flatten(),
            Dense(512, activation='relu'), BatchNormalization(), Dropout(0.5),
            Dense(256, activation='relu'), Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def create_feature_extractor(self):
        """ResNet50-based feature extractor with Global Average Pooling"""
        base = tf.keras.applications.ResNet50(
            include_top=False, 
            weights='imagenet', 
            input_shape=(*self.img_size, 3),
            pooling='avg'  # This gives us 2048 features instead of 32768
        )
        base.trainable = False
        return base

    def create_lstm_model(self, feature_size):
        """LSTM model for sequences"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, feature_size)),
            BatchNormalization(), Dropout(0.3),
            LSTM(32, return_sequences=True),
            BatchNormalization(), Dropout(0.3),
            LSTM(16), BatchNormalization(), Dropout(0.3),
            Dense(32, activation='relu'), Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_cnn(self, images, labels):
        """Train CNN"""
        print("Training CNN model...")
        X_train, X_val, y_train, y_val = train_test_split(images, labels, stratify=labels, test_size=0.2, random_state=42)
        model = self.create_cnn_model()
        callbacks = [EarlyStopping(patience=10, restore_best_weights=True),
                     ReduceLROnPlateau(factor=0.5, patience=5),
                     ModelCheckpoint('models/mizhi_cnn_final.keras', save_best_only=True)]
        os.makedirs('models', exist_ok=True)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=50, batch_size=32, callbacks=callbacks, verbose=1)
        test_loss, test_acc = model.evaluate(X_val, y_val)
        print(f"CNN Test Accuracy: {test_acc:.4f}")
        return model, history

    def prepare_lstm_data(self, images, labels, batch_size=32):
        """Extract features in smaller batches and save to disk"""
        print("Creating feature extractor...")
        feat_ext = self.create_feature_extractor()
        
        os.makedirs('features', exist_ok=True)
        feature_files = []
        
        print(f"Extracting features in batches of {batch_size}...")
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Extract features
            feats = feat_ext.predict(batch, verbose=0)
            
            # Save features and labels
            feat_fname = f'features/feats_{i//batch_size}.npy'
            label_fname = f'features/labels_{i//batch_size}.npy'
            
            np.save(feat_fname, feats)
            np.save(label_fname, batch_labels)
            
            feature_files.append((feat_fname, label_fname))
            
            if (i//batch_size + 1) % 10 == 0:
                print(f"Processed {i//batch_size + 1} batches...")
                
        print(f"Features saved in {len(feature_files)} files.")
        return feature_files

    def create_lstm_sequences(self, feature_files, seq_len=30, max_sequences=1000):
        """Create LSTM sequences from feature files with memory management"""
        print(f"Creating LSTM sequences (max {max_sequences})...")
        
        sequences = []
        seq_labels = []
        sequence_count = 0
        
        for feat_file, label_file in feature_files:
            if sequence_count >= max_sequences:
                break
                
            feats = np.load(feat_file)
            labels = np.load(label_file)
            
            # Create sequences from this batch
            for i in range(len(feats) - seq_len + 1):
                if sequence_count >= max_sequences:
                    break
                    
                seq = feats[i:i+seq_len]
                label = labels[i+seq_len-1]  # Use last label in sequence
                
                sequences.append(seq)
                seq_labels.append(label)
                sequence_count += 1
                
                if sequence_count % 100 == 0:
                    print(f"Created {sequence_count} sequences...")
        
        return np.array(sequences), np.array(seq_labels)

    def train_lstm(self, images, labels):
        """Train LSTM with memory optimization"""
        print("Preparing LSTM data...")
        
        # Extract features and save to disk
        feature_files = self.prepare_lstm_data(images, labels, batch_size=32)
        
        # Create sequences with limited memory usage
        sequences, seq_labels = self.create_lstm_sequences(feature_files, 
                                                         seq_len=self.sequence_length,
                                                         max_sequences=2000)  # Limit sequences
        
        if len(sequences) == 0:
            print("No sequences created. Check your data.")
            return None, None
            
        print(f"Created {len(sequences)} sequences for LSTM training")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, seq_labels, test_size=0.2, random_state=42, stratify=seq_labels
        )
        
        # Create and train model
        model = self.create_lstm_model(self.feature_size)
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint('models/mizhi_lstm_final.keras', save_best_only=True)
        ]
        
        print("Training LSTM model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=16,  # Smaller batch size for memory
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_val, y_val)
        print(f"LSTM Test Accuracy: {test_acc:.4f}")
        
        # Clean up memory
        del sequences, seq_labels, X_train, X_val, y_train, y_val
        gc.collect()
        
        return model, history

    def train_all_models(self, max_images=5000, skip_cnn=False):
        print("Starting MIZHI training...")
        images, labels = self.load_images(max_images)
        
        if len(images) == 0:
            print("No images found, check your dataset path.")
            return
        
        # CNN Training
        if not skip_cnn:
            print("Training CNN...")
            cnn_model, cnn_hist = self.train_cnn(images, labels)
        else:
            print("Skipping CNN training...")
            # Check for different model file extensions
            model_files = [
                'models/mizhi_cnn_final.keras',
                'models/mizhi_cnn_final.h5',
                'models/mizhi_cnn_best.h5',
                'models/mizhi_cnn_model.h5'
            ]
            
            cnn_model = None
            for model_file in model_files:
                if os.path.exists(model_file):
                    print(f"Loading existing CNN model: {model_file}")
                    cnn_model = load_model(model_file)
                    break
            
            if cnn_model is None:
                print("Warning: No existing CNN model found!")
                print("Available files:", [f for f in os.listdir('models') if f.endswith(('.h5', '.keras'))])
                cnn_model = None
        
        # LSTM Training
        print("Training LSTM...")
        lstm_model, lstm_hist = self.train_lstm(images, labels)
        
        print("Training process complete.")
        
        # Clean up features directory if needed
        try:
            import shutil
            if os.path.exists('features') and len(os.listdir('features')) > 100:
                print("Cleaning up feature files...")
                shutil.rmtree('features')
                os.makedirs('features', exist_ok=True)
        except:
            pass

def main():
    DATASET_PATH = input("Enter your dataset path (or press Enter for './dataset'): ").strip() or './dataset'
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path '{DATASET_PATH}' not found.")
        return
    
    # Check if CNN model exists (check multiple extensions)
    model_files = [
        'models/mizhi_cnn_final.keras',
        'models/mizhi_cnn_final.h5',
        'models/mizhi_cnn_best.h5',
        'models/mizhi_cnn_model.h5'
    ]
    
    cnn_exists = any(os.path.exists(f) for f in model_files)
    
    if cnn_exists:
        existing_model = next((f for f in model_files if os.path.exists(f)), None)
        print(f"Found existing CNN model: {existing_model}")
        skip_cnn = input("Skip CNN training? (y/n, default=y): ").strip().lower()
        skip_cnn = skip_cnn != 'n'
    else:
        skip_cnn = False
        print("No existing CNN model found. Will train CNN first.")
    
    os.makedirs('models', exist_ok=True)
    
    trainer = MIZHITrainer(DATASET_PATH)
    trainer.train_all_models(max_images=5000, skip_cnn=skip_cnn)

if __name__ == "__main__":
    main()