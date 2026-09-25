import streamlit as st
import joblib
from preprocessing import clean_text, clean_series  # needed so joblib can unpickle the pipeline

st.set_page_config(page_title="Spam Checker", page_icon="📧", layout="centered")


@st.cache_resource
def load_pipeline():
    return joblib.load('spam_pipeline.pkl')


pipeline = load_pipeline()

st.title("📧 Spam Checker")
st.write("Paste in an email and the model will tell you whether it looks like spam.")

email_text = st.text_area("Email content", height=220, placeholder="Paste the email text here...")

if st.button("Check email", type="primary"):
    if not email_text.strip():
        st.warning("Paste some email text first.")
    else:
        prediction = pipeline.predict([email_text])[0]
        is_spam = prediction == 1

        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba([email_text])[0]
            confidence = round(max(proba) * 100, 1)
        else:
            confidence = None

        if is_spam:
            st.error(f"**Spam**" + (f" — {confidence}% confidence" if confidence else ""))
        else:
            st.success(f"**Not spam**" + (f" — {confidence}% confidence" if confidence else ""))

st.divider()
st.caption("Built with scikit-learn + Streamlit")