import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="crop_disease.tflite")
interpreter.allocate_tensors()

print("TFLite model loaded successfully!")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input:", input_details)
print("Output:", output_details)