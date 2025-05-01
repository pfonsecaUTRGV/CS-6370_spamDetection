# spam_app.py

import streamlit as st
import joblib

# Load the model
model = joblib.load('spam_detector.pkl')

# Streamlit UI
#st.title("📧 CS6730 Spam Detector")
st.title(" CS6730 Spam Detector")
st.write("Enter the content of an email to check if it's spam:")

email_input = st.text_area("Email content:")

if st.button("Detect Spam"):
    if email_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([email_input])[0]
        label = "Spam" if prediction == 1 else "Not Spam"
        #label = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
        if prediction ==1:
            st.warning(f"Email: {label}")
        else:
            st.success(f"Email: {label}")

