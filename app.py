import os
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

st.title("MNIST Digit Predictor")
st.write("Upload an image of a handwritten digit to get a prediction.")


# ใช้ @st.cache_resource เพื่อโหลดโมเดลเพียงครั้งเดียว ไม่ต้องโหลดใหม่ทุกครั้งที่มีการกดปุ่ม
@st.cache_resource
def load_my_model(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None


model_path = "67102010511_mnist_model.keras"
model = load_my_model(model_path)

if model is None:
    st.error(
        f"Model file '{model_path}' not found. Please ensure the model is saved correctly."
    )
else:
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # 1. อ่านรูปภาพจากไฟล์ที่อัปโหลด
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            st.write("")
            st.write("Classifying...")

            # 2. แปลงเป็นขาวดำ (แก้จาก img.convert เป็น image.convert)
            img = image.convert("L")

            # 3. ย่อขนาดภาพเป็น 28x28 พิกเซล
            img = img.resize((28, 28))

            # 4. แปลงเป็น NumPy Array
            img_array = np.array(img)

            # 5. Normalize ค่าพิกเซลจาก [0, 255] ให้อยู่ในช่วง [0, 1]
            img_array = img_array.astype("float32") / 255.0

            # 6. ปรับ Shape ให้เข้ากับ Input ของโมเดล (1, 28, 28)
            img_array = img_array.reshape(1, 28, 28)

            # 7. ทำนายผล
            prediction = model.predict(img_array)

            # 8. หาคลาสที่มีค่าความน่าจะเป็นสูงสุด
            predicted_digit = np.argmax(prediction)

            st.success(
                f"The model predicts the digit is: **{predicted_digit}**"
            )

        except Exception as e:
            st.error(
                f"An error occurred during prediction: {e}. Please ensure the uploaded image is valid and the model is correctly loaded."
            )
