"""
TensorFlow Model Training and Inference
Trains a neural network for car plate detection using transfer learning
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import cv2
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)

logger = logging.getLogger(__name__)


class PlateDetectionModel:
    """
    TensorFlow model for car plate detection
    Uses transfer learning with MobileNetV2 backbone
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (640, 480, 3),
        backbone: str = "mobilenetv2",
        num_classes: int = 1,
        model_path: Optional[str] = None
    ):
        """
        Initialize plate detection model
        
        Args:
            input_shape: Input image shape (height, width, channels)
            backbone: Backbone architecture ('mobilenetv2' or 'efficientnet')
            num_classes: Number of detection classes
            model_path: Path to pre-trained model weights
        """
        self.input_shape = input_shape
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.model_path = model_path
        self.model: Optional[keras.Model] = None
        self.history: Optional[Dict] = None
        self.is_trained = False
        
        logger.info(f"Initializing PlateDetectionModel with {backbone} backbone")
    
    def _create_backbone(self) -> keras.Model:
        """
        Create feature extraction backbone using transfer learning
        
        Returns:
            Keras model for feature extraction
        """
        if self.backbone_name.lower() == "mobilenetv2":
            logger.info("Loading MobileNetV2 backbone")
            backbone = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif self.backbone_name.lower() == "efficientnet":
            logger.info("Loading EfficientNetB0 backbone")
            backbone = EfficientNetB0(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")
        
        # Freeze backbone weights for transfer learning
        backbone.trainable = False
        return backbone
    
    def build(self) -> keras.Model:
        """
        Build the complete detection model
        Outputs: bounding boxes, confidence scores, and class predictions
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building detection model...")
        
        # Input layer
        inputs = keras.Input(shape=self.input_shape)
        
        # Backbone (feature extraction)
        backbone = self._create_backbone()
        features = backbone(inputs)
        
        # Global average pooling
        pooled = layers.GlobalAveragePooling2D()(features)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(pooled)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output heads
        # 1. Confidence score (plate presence)
        confidence = layers.Dense(1, activation='sigmoid', name='confidence')(x)
        
        # 2. Bounding box (4 values: ymin, xmin, ymax, xmax)
        bbox = layers.Dense(4, activation='sigmoid', name='bbox')(x)
        
        # 3. Class prediction
        class_pred = layers.Dense(self.num_classes, activation='softmax', name='class')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=[confidence, bbox, class_pred])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss={
                'confidence': BinaryCrossentropy(),
                'bbox': MeanSquaredError(),
                'class': 'categorical_crossentropy'
            },
            loss_weights={'confidence': 1.0, 'bbox': 0.5, 'class': 0.5},
            metrics={
                'confidence': ['accuracy'],
                'bbox': ['mae'],
                'class': ['accuracy']
            }
        )
        
        self.model = model
        logger.info("Model built successfully")
        logger.info(f"Total parameters: {model.count_params():,}")
        return model
    
    def summary(self) -> None:
        """Print model summary"""
        if self.model is None:
            logger.error("Model not built yet. Call build() first.")
            return
        self.model.summary()
    
    def train(
        self,
        train_data: Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        val_data: Optional[Tuple] = None,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            train_data: Tuple of (X_train, (y_confidence, y_bbox, y_class))
            val_data: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        X_train, (y_conf, y_bbox, y_class) = train_data
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                ModelCheckpoint(
                    filepath='best_model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
        
        # Train model
        history = self.model.fit(
            X_train,
            {'confidence': y_conf, 'bbox': y_bbox, 'class': y_class},
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        self.is_trained = True
        logger.info("Training completed")
        
        return self.history
    
    def evaluate(
        self,
        test_data: Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            test_data: Tuple of (X_test, (y_confidence, y_bbox, y_class))
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        logger.info("Evaluating model...")
        
        X_test, (y_conf, y_bbox, y_class) = test_data
        
        results = self.model.evaluate(
            X_test,
            {'confidence': y_conf, 'bbox': y_bbox, 'class': y_class},
            verbose=0
        )
        
        eval_dict = {
            'total_loss': results[0],
            'confidence_loss': results[1],
            'bbox_loss': results[2],
            'class_loss': results[3],
            'confidence_accuracy': results[4],
            'bbox_mae': results[5],
            'class_accuracy': results[6]
        }
        
        logger.info(f"Evaluation Results: {eval_dict}")
        return eval_dict
    
    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on an image
        
        Args:
            image: Input image (BGR or RGB, will be converted)
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        # Preprocess image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input shape
        try:
            image_resized = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        except Exception as e:
            logger.error(f"Resize failed: {e}. Image shape: {image.shape}, Target: {self.input_shape}")
            raise
        
        # Normalize
        image_normalized = image_resized.astype('float32') / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        # Predict with error handling
        try:
            confidence, bbox, class_pred = self.model.predict(image_batch, verbose=0)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}. Input shape: {image_batch.shape}, Expected: (1, {self.input_shape[0]}, {self.input_shape[1]}, {self.input_shape[2]})")
            raise
        
        return {
            'confidence': confidence[0][0],
            'bbox': bbox[0],
            'class': class_pred[0],
            'class_index': np.argmax(class_pred[0])
        }
    
    def save(self, save_path: str) -> None:
        """
        Save model to disk
        
        Args:
            save_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        logger.info(f"Saving model to {save_path}")
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        self.model.save(str(save_dir / "model.h5"))
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(save_dir / "model.json", "w") as f:
            f.write(model_json)
        
        # Save config
        config = {
            'input_shape': self.input_shape,
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'is_trained': self.is_trained,
            'saved_at': datetime.now().isoformat()
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved successfully at {save_path}")
    
    def load(self, load_path: str) -> None:
        """
        Load model from disk
        
        Args:
            load_path: Path to load the model from
        """
        logger.info(f"Loading model from {load_path}")
        
        load_dir = Path(load_path)
        
        # Load model
        self.model = keras.models.load_model(
            str(load_dir / "model.h5"),
            compile=True
        )
        
        # Load config
        with open(load_dir / "config.json", "r") as f:
            config = json.load(f)
            self.input_shape = tuple(config['input_shape'])
            self.backbone_name = config['backbone']
            self.num_classes = config['num_classes']
            self.is_trained = config.get('is_trained', False)
        
        logger.info("Model loaded successfully")


class DataPreprocessor:
    """Utilities for preprocessing training data"""
    
    @staticmethod
    def load_image(image_path: str, size: Tuple[int, int]) -> np.ndarray:
        """
        Load and preprocess an image
        
        Args:
            image_path: Path to image file
            size: Target size (height, width)
            
        Returns:
            Normalized image array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size[1], size[0]))
        image = image.astype('float32') / 255.0
        
        return image
    
    @staticmethod
    def augment_image(image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to image
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Random rotation
        angle = np.random.randint(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Random brightness
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 1)
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1)
        
        return image


class ModelTrainer:
    """End-to-end model training pipeline"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize trainer
        
        Args:
            model_config: Configuration dictionary for model
        """
        self.config = model_config
        self.model: Optional[PlateDetectionModel] = None
        logger.info(f"ModelTrainer initialized with config: {model_config}")
    
    def create_model(self) -> PlateDetectionModel:
        """Create and build model"""
        self.model = PlateDetectionModel(
            input_shape=tuple(self.config.get('input_shape', (640, 480, 3))),
            backbone=self.config.get('backbone', 'mobilenetv2'),
            num_classes=self.config.get('num_classes', 1)
        )
        self.model.build()
        return self.model
    
    def train_from_scratch(
        self,
        X_train: np.ndarray,
        y_train: Tuple[np.ndarray, np.ndarray, np.ndarray],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Tuple] = None,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Train model from scratch
        
        Args:
            X_train: Training images
            y_train: Tuple of (confidence, bbox, class) labels
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training results
        """
        if self.model is None:
            self.create_model()
        
        train_data = (X_train, y_train)
        val_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.train(
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return history
    
    def save_model(self, save_path: str) -> None:
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(save_path)
    
    def load_model(self, load_path: str) -> None:
        """Load trained model"""
        self.model = PlateDetectionModel()
        self.model.load(load_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example: Create and build model
    logger.info("Creating plate detection model...")
    model = PlateDetectionModel(
        input_shape=(640, 480, 3),
        backbone='mobilenetv2',
        num_classes=1
    )
    model.build()
    model.summary()
    
    # Example: Create dummy training data
    logger.info("Creating dummy training data...")
    X_train = np.random.rand(100, 640, 480, 3).astype('float32')
    y_confidence = np.random.rand(100, 1).astype('float32')
    y_bbox = np.random.rand(100, 4).astype('float32')
    y_class = np.eye(1)[np.random.randint(0, 1, 100)]
    
    # Example: Train model
    logger.info("Training model...")
    history = model.train(
        train_data=(X_train, (y_confidence, y_bbox, y_class)),
        epochs=3,
        batch_size=32
    )
    
    logger.info("Training complete!")
