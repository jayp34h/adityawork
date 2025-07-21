import streamlit as st
import pickle
import numpy as np

# Load trained model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# App title
st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("📨 Spam Email Classifier")
st.markdown("Enter an email message below, and the model will classify it as **Spam** or **Ham**.")

# User input
user_input = st.text_area("✉️ Type your email message here:", height=150)

# Predict button
if st.button("🔍 Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some message.")
    else:
        # Vectorize and predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)

        # Show result
        if prediction[0] == 1:
            st.success("✅ This is a **Ham** (Not Spam) message.")
        else:
            st.error("🚫 This is a **Spam** message.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
