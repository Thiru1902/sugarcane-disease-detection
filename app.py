import streamlit as st
import os
from groq import Groq
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Sugarcane Disease Detection",
    page_icon="🌾",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.stApp { max-width: 1200px; margin: 0 auto; }
h1 { color: #2d5016; text-align: center; font-weight: 600; margin-bottom: 0.5rem; }
.subtitle { text-align: center; color: #5a7a3d; font-size: 1.1rem; margin-bottom: 2rem; }
.input-card { background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem; }
.result-container { background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-top: 2rem; }
.disease-name { color: #2d5016; font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem; }
.section-header { color: #5a7a3d; font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem; }
.disclaimer { text-align: center; color: #6c757d; font-size: 0.9rem; margin-top: 3rem; padding: 1rem; background-color: #fff3cd; border-radius: 5px; }
.image-container { border: 2px solid #e9ecef; border-radius: 10px; padding: 1rem; background-color: white; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ------------------ Helper functions ------------------

def initialize_groq_client():
    """Initialize Groq client"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ Groq API key not found. Please set GROQ_API_KEY in your environment.")
        st.stop()
    return Groq(api_key=api_key)

def encode_image(image):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_image_input():
    """Get image from camera or upload"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("### 📷 Capture Image")
        camera_image = st.camera_input("Use your device camera")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Upload Image")
        uploaded_image = st.file_uploader(
            "Choose from gallery",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if camera_image:
        return Image.open(camera_image)
    elif uploaded_image:
        return Image.open(uploaded_image)
    return None

def analyze_image_with_groq(client, image):
    """Send image to Groq model for analysis"""
    base64_image = encode_image(image)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """
You are an expert agricultural AI assistant specializing in sugarcane diseases.
Analyze this sugarcane leaf image and provide:

1. Disease Identification: Name the most likely disease (use terms like "likely", "possibly")
2. Visible Symptoms: List 3-4 key visual symptoms
3. Recommended Actions: Suggest 2-3 general preventive or treatment measures

If this is not a sugarcane leaf, politely indicate that.
"""},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_completion_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during analysis: {str(e)}"

def display_results(analysis_text):
    """Parse and display results neatly"""
    st.markdown('<div class="result-container">', unsafe_allow_html=True)

    lines = analysis_text.split('\n')
    disease_name, symptoms, actions = "", [], []
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "disease" in line.lower() and "identification" in line.lower():
            current_section = "disease"
            continue
        elif "symptom" in line.lower():
            current_section = "symptoms"
            continue
        elif "action" in line.lower() or "treatment" in line.lower() or "recommend" in line.lower():
            current_section = "actions"
            continue

        if current_section == "disease" and line and not disease_name:
            disease_name = line.replace('*', '').replace('#', '').strip()
        elif current_section == "symptoms" and line:
            if line.startswith('-') or line.startswith('•') or line[0].isdigit():
                symptoms.append(line.lstrip('-•0123456789. '))
        elif current_section == "actions" and line:
            if line.startswith('-') or line.startswith('•') or line[0].isdigit():
                actions.append(line.lstrip('-•0123456789. '))

    if disease_name:
        st.markdown(f'<div class="disease-name">🔍 {disease_name}</div>', unsafe_allow_html=True)

    if symptoms:
        st.markdown('<div class="section-header">📋 Visible Symptoms:</div>', unsafe_allow_html=True)
        for s in symptoms:
            st.markdown(f"• {s}")

    if actions:
        st.markdown('<div class="section-header">💡 Recommended Actions:</div>', unsafe_allow_html=True)
        for a in actions:
            st.markdown(f"• {a}")

    if not disease_name and not symptoms and not actions:
        st.markdown(analysis_text)

    st.info("ℹ️ **AI-generated advisory** - For educational purposes only")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Main ------------------

def main():
    st.title("🌾 Sugarcane Leaf Disease Detection System")
    st.markdown('<p class="subtitle">AI-based visual analysis for early disease identification</p>', unsafe_allow_html=True)

    client = initialize_groq_client()
    image = get_image_input()

    if image:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="Captured/Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing image... Please wait"):
                analysis = analyze_image_with_groq(client, image)
                display_results(analysis)
    else:
        st.info("👆 Please capture or upload a sugarcane leaf image to begin analysis")

    st.markdown(
        '<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This system is for educational and advisory purposes only. '
        'For critical agricultural decisions, consult qualified agricultural experts.</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
