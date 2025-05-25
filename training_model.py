import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Set dataset paths
BASE_DIR = "C:\\Users\\HARISH\\OneDrive\\Desktop\\ML PROJECTS\\brain_tumor_app"  
TRAIN_DIR = os.path.join(BASE_DIR, "datasets\\Training")
TEST_DIR = os.path.join(BASE_DIR, "datasets\\Testing")

# Image Augmentation for Training Set
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only Rescale for Test Set (No Augmentation)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load Training Data
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load Test Data
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Define Improved CNN Model
model = Sequential([
    # First Conv Block
    Conv2D(32, (3, 3), padding='same', input_shape=(224, 224, 3)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(2, 2),

    # Second Conv Block
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(2, 2),

    # Third Conv Block
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(2, 2),

    # Fourth Conv Block
    Conv2D(256, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(2, 2),

    # Fifth Conv Block (Added for deeper learning)
    Conv2D(512, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(2, 2),

    Flatten(),
    
    # Fully Connected Layers
    Dense(512),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),

    Dense(256),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),

    Dense(4, activation='softmax')  # 4 classes: Glioma, Pituitary, Meningioma, No Tumor
])

# Compile Model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Print Model Summary
model.summary()

# Define Callbacks for Better Training
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    "C:\\Users\\HARISH\\OneDrive\\Desktop\\ML PROJECTS\\brain_tumor_app\\models\\best_brain_tumor_model_v2.keras", 
    save_best_only=True, 
    monitor='val_accuracy', 
    mode='max'
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train Model
history = model.fit(
    train_data,
    epochs=50,
    validation_data=test_data,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Save Final Model
model.save("C:\\Users\\HARISH\\OneDrive\\Desktop\\ML PROJECTS\\brain_tumor_app\\models\\final_brain_tumor_model_v2.keras")
print("Training complete. Model saved.")
