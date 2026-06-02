import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
# ... the rest of your front.py code
# Importing directly from your newly named chat.py script
from chat import build_vector_store, get_qa_chain

st.set_page_config(page_title="Customer Support QA Chatbot", page_icon="🇲🇲", layout="wide")

st.title("Customer Support Chatbot")
st.caption("Powered by Mistral AI & FAISS. Trained on your localized English and Myanmar FAQ sheets.")

# --- Sidebar Setup ---
with st.sidebar:
    st.header("Setup & Administration")
    mistral_api_key = st.text_input("Mistral API Key:", type="password")
    
    uploaded_files = st.file_uploader(
        "Upload QA Datasets:", 
        type=["xlsx"], 
        accept_multiple_files=True,
        help="Upload ChatBot_Training_668_EN_QA.xlsx and ChatBot_Training_668_MM_QA.xlsx here."
    )
    
    process_btn = st.button("Train Vector Model")

# --- Manage App Session States ---
if "vs_state" not in st.session_state:
    st.session_state.vs_state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Train Button Action ---
if process_btn:
    if not mistral_api_key:
        st.sidebar.error("Please enter your Mistral API Key first.")
    elif not uploaded_files:
        st.sidebar.error("Please select your dataset files.")
    else:
        with st.spinner("Parsing 1,300+ bilingual questions..."):
            try:
                vector_store = build_vector_store(uploaded_files, mistral_api_key)
                st.session_state.vs_state = vector_store
                st.sidebar.success("Database trained successfully!")
            except Exception as e:
                st.sidebar.error(f"Failed to compile database: {e}")

# --- Render Previous Messages ---
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# --- User Input Interaction ---
if user_input := st.chat_input("မေးခွန်းများကို ဤနေရာတွင် မေးမြန်းနိုင်ပါသည်။ / Ask a question here..."):
    # Show user input instantly
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Check if database has been compiled
    if not st.session_state.vs_state:
        with st.chat_message("assistant"):
            warning_text = "Please upload files and click 'Train Vector Model' in the sidebar before querying."
            st.markdown(warning_text)
            st.session_state.chat_history.append({"role": "assistant", "content": warning_text})
    else:
        with st.chat_message("assistant"):
            reply_container = st.empty()
            with st.spinner("Processing intent..."):
                try:
                    # Run RAG execution chain
                    chain = get_qa_chain(st.session_state.vs_state, mistral_api_key)
                    result = chain.invoke({"input": user_input})
                    
                    bot_response = result["answer"]
                    reply_container.markdown(bot_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                except Exception as e:
                    err_msg = f"A processing error occurred: {e}"
                    reply_container.markdown(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
