import streamlit as st
import joblib

# Load model and vectorizer
model, vectorizer = joblib.load("model.pkl")

# App title
st.title("📰 Fake News Detector")
st.write("Enter a news article and let the model tell you if it's **Fake** or **Real**.")

# Text input
user_input = st.text_area("✏️ Paste news article here", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        # Vectorize and predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        # Show result
        if prediction == 1:
            st.success("✅ This news is *Real*.")
        else:
            st.error("🚨 This news is *Fake*.")
