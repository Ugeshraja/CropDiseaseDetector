import tensorflow as tf

print("Loading Model...")

model = tf.keras.models.load_model(
    "trained_model.keras",
    compile=False
)

print("Model Loaded Successfully!")
print("Input Shape:", model.input_shape)
print("Output Shape:", model.output_shape)