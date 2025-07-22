import streamlit as st
import os
import dotenv
from ocr_pipeline import process_and_upload
from chatbot_utils import embed_query, search_pinecone, generate_gpt_reply
from pinecone import Pinecone

def pinecone_index_is_empty(pinecone_api_key, pinecone_index_name):
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    stats = index.describe_index_stats()
    return stats.get("total_vector_count", 0) == 0

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

st.set_page_config(page_title="Pamphlet AI Chatbot", layout="centered")

# ---- Minimal iMessage-like CSS (no chat-container, no extra heading) ----
st.markdown("""
    <style>
    body {
        background-color: #f5f5f7;
    }
    .fixed-title {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background: rgba(240, 242, 246, 1);
        border-bottom: 1px solid #ddd;
        padding: 1rem;
        z-index: 999;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .user-message {
        display: flex;
        justify-content: flex-end;
        text-align: right;
    }
    .bot-message {
        display: flex;
        justify-content: flex-start;
        text-align: left;
    }
    .chat-bubble {
        max-width: 70%;
        padding: 14px 20px;
        border-radius: 22px;
        margin: 8px 0;
        word-wrap: break-word;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .user-bubble {
        background-color: #007aff;
        color: white;
        border-bottom-right-radius: 6px;
        border-top-right-radius: 22px;
        border-top-left-radius: 22px;
        border-bottom-left-radius: 22px;
    }
    .bot-bubble {
        background-color: #e5e5ea;
        color: #222;
        border-bottom-left-radius: 6px;
        border-top-right-radius: 22px;
        border-top-left-radius: 22px;
        border-bottom-right-radius: 22px;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Fixed Title ----
st.markdown('<div class="fixed-title">Pamphlet AI Chatbot</div>', unsafe_allow_html=True)

# ---- Sidebar Navigation ----
if "page" not in st.session_state:
    if pinecone_index_is_empty(pinecone_api_key, pinecone_index_name):
        default_page = "Upload"
    else:
        default_page = "Chat"
else:
    default_page = st.session_state.page

page = st.sidebar.radio(
    "Navigation",
    options=["Chat", "Upload", "Reset Memory"],
    index=["Chat", "Upload", "Reset Memory"].index(default_page)
)
# Do NOT set st.session_state.page = page here!

# ---- Upload Processing Flag ----
if "upload_processed" not in st.session_state:
    st.session_state.upload_processed = False

# ---- Reset Memory Page ----
if page == "Reset Memory":
    reset_confirm = st.sidebar.checkbox("Confirm To Reset Memory", value=False)
    reset_btn = st.sidebar.button("Reset Memory", type="primary", disabled=not reset_confirm)
    if reset_btn and reset_confirm:
        with st.spinner("Resetting memory..."):
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(pinecone_index_name)
            index.delete(delete_all=True)
            st.session_state.chat_history = []
            st.session_state.page = "Upload"
            st.session_state.upload_processed = False  # Reset upload flag
            st.success("All embeddings deleted from Pinecone. Memory reset!")
            st.rerun()

# ---- Upload Widget and Processing (only on upload page) ----
if page == "Upload":
    # st.title("Pamphlet AI Chatbot")
    uploaded_files = st.file_uploader(
        "Upload one or more pamphlet images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )
    if uploaded_files and not st.session_state.upload_processed:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                try:
                    process_and_upload(temp_path, openai_api_key, pinecone_api_key, pinecone_index_name)
                    st.success(f"{uploaded_file.name} processed and uploaded to Pinecone!")
                except Exception as e:
                    st.error(f"Processing failed for {uploaded_file.name}: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        st.session_state.pamphlet_uploaded = True
        # st.session_state.chat_history = [] # To reset chat after upload
        st.session_state.upload_processed = True  # Prevent reprocessing
        # Do NOT change page or rerun here
# Reset the flag when user navigates to upload page
if page == "Upload" and st.session_state.get("upload_processed"):
    st.session_state.upload_processed = False

elif page == "Chat":
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # ---- Minimal chat rendering (no chat-container, no heading) ----
    chat_placeholder = st.container()
    with chat_placeholder:
        for msg in st.session_state.chat_history:
            css_class = "user-message" if msg["role"] == "user" else "bot-message"
            bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
            st.markdown(f"""
                <div class="{css_class}">
                    <div class="chat-bubble {bubble_class}">
                        {msg["content"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        # Show pending bot reply placeholder
        if st.session_state.get("pending_bot_reply"):
            st.markdown('<p><em>Processing...</em></p>', unsafe_allow_html=True)
    # ---- Chat Input (fixed at bottom by Streamlit) ----
    user_input = st.chat_input("Ask a question about your pamphlet...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        st.session_state.pending_bot_reply = user_input.strip()
    # ---- Process Pending Bot Reply ----
    if st.session_state.get("pending_bot_reply"):
        try:
            query_embedding = embed_query(st.session_state.pending_bot_reply)
            results = search_pinecone(query_embedding, top_k=5)
            context = "\n\n".join([m.metadata["text"] for m in results])
            bot_reply = generate_gpt_reply(st.session_state.chat_history, context, st.session_state.pending_bot_reply)
        except Exception as ex:
            bot_reply = f"Error: {ex}"
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        st.session_state.pending_bot_reply = None
        st.rerun() 