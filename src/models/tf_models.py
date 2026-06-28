import tensorflow as tf
from tensorflow.keras import layers, models

def build_thermal_sr_model(input_shape=(256, 256, 1), scale_factor=2):
    """
    TensorFlow/Keras Super-Resolution model for Thermal Infrared Imagery (200m -> 100m).
    """
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    
    # Residual blocks
    res = x
    for _ in range(4):
        y = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(res)
        y = layers.Conv2D(64, kernel_size=3, padding='same')(y)
        res = layers.Add()([res, y])
        
    x = layers.Add()([x, res])
    
    # Upsampling block (PixelShuffle / Conv2DTranspose)
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=scale_factor, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(1, kernel_size=3, padding='same')(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name="ThermalSR_TF")


def build_thermal_colorizer_model(input_shape=(512, 512, 1)):
    """
    TensorFlow/Keras U-Net colorization model (1-channel TIR -> 3-channel RGB).
    Output channels correspond to Layer 1: Blue, Layer 2: Green, Layer 3: Red.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(b)
    
    # Decoder
    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    u2 = layers.concatenate([u2, c2])
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(u2)
    
    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c1])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    
    outputs = layers.Conv2D(3, 1, activation='sigmoid')(c4)
    
    return models.Model(inputs=inputs, outputs=outputs, name="ThermalColorizer_TF")
