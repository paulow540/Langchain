import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="LangChain Essay & Poem Generator", page_icon="📝", layout="centered")

st.title("Essay & Poem Generator")
st.write("This app uses FastAPI + LangChain + OpenAI/Ollama to generate essays and poems.")



st.sidebar.header("Choose Task")
task = st.sidebar.radio("Select what you want to generate:", ["Essay", "Poem"])

topic = st.text_input("Enter a topic:")

if st.button("Generate"):
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Generating..."):
            try:
                if task == "Essay":
                    response = requests.post(f"{API_URL}/essay/invoke", json={"input": {"topic": topic}})
                else:
                    response = requests.post(f"{API_URL}/poem/invoke", json={"input": {"topic": topic}})

                if response.status_code == 200:
                    result = response.json()
                    output = result.get("output", "No output received.")
                    st.success("Here’s the result:")
                    st.write(output)
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

