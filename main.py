import streamlit as st
from pipeline import check_text

st.set_page_config(page_title="Fact Checker", page_icon="🕵️", layout="centered")

st.title("🕵️ Local RAG Fact Checker")
st.write(
    "Enter a short Hindi or English news/social media statement. "
    "The system will extract claims, retrieve relevant facts, and classify them."
)

user_text = st.text_area(
    "Input text:",
    height=150,
    placeholder="केंद्रीय मंत्री श्री मनोहर लाल स्वच्छता ही सेवा (एसएचएस)-श्रमदान में भाग लेंगे।"
)

if st.button("Check facts"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            result = check_text(user_text)

        st.subheader("Results")

        for r in result["results"]:
            st.markdown("---")
            st.markdown(f"**Claim:** {r['claim']}")
            st.markdown(f"**Verdict:** {r['emoji_verdict']}")
            st.markdown(f"**Reasoning:** {r['reasoning']}")

            if r["evidence"]:
                with st.expander("Evidence"):
                    for i, ev in enumerate(r["evidence"], start=1):
                        st.write(f"**{i}.** {ev}")
