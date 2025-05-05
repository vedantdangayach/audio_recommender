import tensorflow as tf
from tensorflow.keras import layers, Model

class AudioEncoder(Model):
    def __init__(self, latent_dim=128):
        super(AudioEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(128, 128, 1)),  # Mel spectrogram input
            
            # First conv block
            layers.Conv2D(32, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Second conv block
            layers.Conv2D(64, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Third conv block
            layers.Conv2D(128, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Fourth conv block
            layers.Conv2D(256, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Flatten and dense to latent space
            layers.Flatten(),
            layers.Dense(latent_dim)
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(8 * 8 * 256),
            layers.Reshape((8, 8, 256)),
            
            # First transpose conv block
            layers.Conv2DTranspose(128, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Second transpose conv block
            layers.Conv2DTranspose(64, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Third transpose conv block
            layers.Conv2DTranspose(32, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Final transpose conv block
            layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
