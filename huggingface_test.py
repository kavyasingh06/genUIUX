import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# ---------------------------
# Hugging Face Authentication
# ---------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]  # Token from Streamlit Secrets
login(HF_TOKEN)  # Authenticate to access gated StarCoder

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="💡 Generative UI/UX Designer",
    page_icon="🎨",
    layout="wide"
)

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.title("Settings ⚙️")
framework = st.sidebar.selectbox("Choose Framework", ["React", "Flutter"])
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=500, value=300)
temperature = st.sidebar.slider("Creativity (Temperature)", min_value=0.1, max_value=1.0, value=0.7)

# ---------------------------
# Page Title
# ---------------------------
st.title("💡 Generative UI/UX Designer")
st.write("Type a prompt and generate UI code for **React** or **Flutter** instantly!")

# ---------------------------
# Load StarCoder Model
# ---------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder", use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder", use_auth_token=HF_TOKEN)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # -1 = CPU, 0 = GPU
    )
    return generator

generator = load_model()

# ---------------------------
# Prompt Input
# ---------------------------
prompt = st.text_area("Enter your UI/UX prompt here:")

# ---------------------------
# Generate Code Button
# ---------------------------
if st.button("Generate Code 🚀"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt first!")
    else:
        with st.spinner("Generating code..."):
            output = generator(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature
            )
            code = output[0]['generated_text']

        # Display Output
        tab1, tab2 = st.tabs(["Code Preview", "Download"])
        with tab1:
            st.code(code, language="dart" if framework=="Flutter" else "javascript")
        with tab2:
            filename = "ui_code.dart" if framework=="Flutter" else "ui_code.js"
            st.download_button(
                label=f"Download {framework} Code",
                data=code,
                file_name=filename,
                mime="text/plain"
            )

# Footer
st.markdown(
    "<p style='text-align:center;color:gray;'>Built with 💡 Streamlit + Hugging Face StarCoder</p>",
    unsafe_allow_html=True
)
