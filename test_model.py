import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

try:
    model = tf.keras.models.load_model(
        "brain_tumor_model.h5",
        compile=False
    )

    print("✅ Model loaded successfully!")
    model.summary()

except Exception as e:
    print("\n❌ Failed to load model")
    print(type(e).__name__)
    print(e)