import streamlit as st
import time

from sts_inference import sts_inference

st.title("Sentence Similarity")

sentence1 = st.text_input("First sentence")
sentence2 = st.text_input("Second sentence")

if st.button("Calculate"):
    if sentence1 == "":
        st.warning("Input sentence1")
    elif sentence2 == "":
        st.warning("Input sentence2")
    else:
        placeholder = st.empty()

        similarity = None

        with st.spinner("Wait for calculating how close"):
            similarity = sts_inference(sentence1, sentence2)

        with st.container():
            placeholder.success(f"Two sentences are {similarity}% similar.")
