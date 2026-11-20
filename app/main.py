"""Streamlit entry point: `streamlit run app/main.py`"""
import streamlit as st

from app.classify import ModelNotFoundError, load_classifier
from app.core import milestone_5
from app.ui_logic import EXAMPLES

st.set_page_config(page_title="Customer Feedback Analyzer")


@st.cache_resource(show_spinner="Loading the fine-tuned model...")
def get_classifier():
    return load_classifier()  # never trains here


st.title("Customer Feedback Analyzer")
st.write("Retail managers use this tool to automatically categorize customer feedback comments "
         "(e.g., \"product issue\", \"staff praise\") to quickly identify trends and prioritize improvements. "
         "A fine-tuned GPT-2 scores one special label token per category.")

try:
    classifier = get_classifier()
except ModelNotFoundError:
    st.error("No trained model found. Train one first (about a minute on an Apple Silicon Mac), "
             "then reload this page:")
    st.code("python -m app.cli dataset\npython -m app.cli train", language="bash")
    st.stop()

st.session_state.setdefault("comment", "")
st.caption("Try an example:")
for col, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    if col.button(example[:22] + "...", help=example, use_container_width=True):
        st.session_state["comment"] = example

comment = st.text_area("Customer comment", key="comment", height=120)
if st.button("Classify", type="primary"):
    if not comment.strip():
        st.warning("Type or pick a comment first.")
    else:
        result = milestone_5(comment, classifier)
        st.subheader(result["display"])
        st.caption(f"{result['description']} - confidence {result['confidence']:.1%}")
        for name, score in result["ranked"]:
            st.progress(score, text=f"{name}: {score:.1%}")
