import torch
import numpy as np
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Toxicity Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 30%, #764ba2 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: opaque;
        background-clip: text;
        text-align: center;
        letter-spacing: -0.02em;
        margin-bottom: 2rem;
        text-shadow: none;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .toxic-severe {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .toxic-moderate {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .non-toxic {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Model Paths - HateBERT only
MODEL_PATHS = {
    "hatebert": "models/best_hatebert",
}

# Alternative demo models (fallback)
DEMO_MODEL_PATHS = {
    "unitary-toxic-bert": "unitary/toxic-bert",
}

# Label Mapping
LABEL_MAPPING = {0: "Severely Toxic", 1: "Moderately Toxic", 2: "Non-Toxic"}
LABEL_COLORS = {
    "Severely Toxic": "#f44336",
    "Moderately Toxic": "#ff9800", 
    "Non-Toxic": "#4caf50"
}

@st.cache_resource
def load_models():
    """Load HateBERT model and tokenizer - cached for performance"""
    models = {}
    tokenizers = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Progress bar for loading
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Check if local HateBERT model is available, otherwise use demo model
    local_model_available = False
    try:
        # Try to load tokenizer to check if model exists
        AutoTokenizer.from_pretrained(MODEL_PATHS["hatebert"])
        local_model_available = True
    except:
        local_model_available = False
    
    if local_model_available:
        path_to_use = MODEL_PATHS["hatebert"]
        st.info("🔄 Loading local HateBERT model...")
    else:
        path_to_use = list(DEMO_MODEL_PATHS.values())[0]
        st.info("🔄 Local HateBERT model not found, loading demo model from Hugging Face...")
    
    # Load the model
    status_text.text("Loading HateBERT model...")
    try:
        tokenizers["hatebert"] = AutoTokenizer.from_pretrained(path_to_use)
        models["hatebert"] = AutoModelForSequenceClassification.from_pretrained(path_to_use).to(device).eval()
        st.success("✅ Loaded HateBERT model")
        loaded_count = 1
    except Exception as e:
        st.error(f"❌ Failed to load HateBERT model: {str(e)}")
        loaded_count = 0
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ Successfully loaded {loaded_count}/1 models")
    
    return models, tokenizers, device

def classify_comment(comment, models, tokenizers, device, lime_mode=False):
    """Classify a comment using HateBERT model"""
    if not models or "hatebert" not in models:
        return "No HateBERT model loaded", np.array([0.33, 0.33, 0.34]), {}
    
    tokenizer = tokenizers["hatebert"]
    model = models["hatebert"]
    
    inputs = tokenizer(
        comment, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    
    # Handle different output shapes
    if len(probs) == 2:  # Binary classification
        # Convert to 3-class: [non-toxic, moderately-toxic, severely-toxic]
        probs = np.array([probs[0], probs[1] * 0.5, probs[1] * 0.5])
    elif len(probs) > 3:
        # Take first 3 classes
        probs = probs[:3]
        probs = probs / np.sum(probs)  # Renormalize
    
    model_probs = {"hatebert": probs}
    
    if lime_mode:
        return probs
    
    predicted_label = np.argmax(probs)
    return LABEL_MAPPING[predicted_label], probs, model_probs



def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">Abusive Comment Detection System</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This system uses the HateBERT model to classify text toxicity levels.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Model loading status
        with st.expander("🔧 Model Status", expanded=True):
            if 'models_loaded' not in st.session_state:
                st.session_state.models_loaded = False
            
            if not st.session_state.models_loaded:
                if st.button("🚀 Load HateBERT Model", type="primary"):
                    with st.spinner("Loading HateBERT model..."):
                        models, tokenizers, device = load_models()
                        st.session_state.models = models
                        st.session_state.tokenizers = tokenizers
                        st.session_state.device = device
                        st.session_state.models_loaded = True
                        st.rerun()
            else:
                st.success("✅ HateBERT model loaded")
                if st.button("🔄 Reload Model"):
                    st.session_state.models_loaded = False
                    st.rerun()
        
        # Settings
        st.header("⚙️ Settings")
        show_model_details = st.checkbox("Show HateBERT model probabilities", value=True)
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Classes:**
            - 🔴 **Severely Toxic**: Highly offensive content
            - 🟡 **Moderately Toxic**: Somewhat inappropriate content  
            - 🟢 **Non-Toxic**: Clean, appropriate content
            """)
    
    # Main content area
    if not st.session_state.get('models_loaded', False):
        st.warning("⚠️ Please load the HateBERT model from the sidebar to start analyzing text.")
        st.stop()
    
    # Text input
    st.header("📝 Enter Text for Analysis")
    
    # Example buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("😊 Positive Example"):
            st.session_state.input_text = "You're such an amazing person! Keep up the great work!"
    with col2:
        if st.button("😐 Neutral Example"):
            st.session_state.input_text = "I disagree with your opinion, but I respect your right to have it"
    with col3:
        if st.button("😠 Moderate Example"):
            st.session_state.input_text = "This is completely stupid and makes no sense"
    with col4:
        if st.button("🚨 Severe Example"):
            st.session_state.input_text = "I hate you and everything you stand for, you worthless piece of trash"
    
    # Text input area
    user_input = st.text_area(
        "Enter your text here:",
        value=st.session_state.get('input_text', ''),
        height=150,
        placeholder="Type or paste the comment you want to analyze..."
    )
    
    # Analysis button
    if st.button("🔍 Analyze Text", type="primary", disabled=not user_input.strip()):
        if user_input.strip():
            with st.spinner("🧠 Analyzing text..."):
                # Get prediction
                prediction, ensemble_probs, model_probs = classify_comment(
                    user_input, 
                    st.session_state.models,
                    st.session_state.tokenizers,
                    st.session_state.device
                )
                
                # Display results
                st.header("📊 Analysis Results")
                
                # Main prediction with colored box
                prediction_class = "not at all toxic"
                st.markdown(f"""
                <div 
                # class="prediction-box {prediction_class}">
                    <h2>Prediction: {prediction}</h2>
                    <p>Confidence: {max(ensemble_probs):.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics row
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Severely Toxic",
                        f"{ensemble_probs[0]:.3f}",
                        delta=f"{ensemble_probs[0] - 0.333:.3f}"
                    )
                with col2:
                    st.metric(
                        "Moderately Toxic", 
                        f"{ensemble_probs[1]:.3f}",
                        delta=f"{ensemble_probs[1] - 0.333:.3f}"
                    )
                with col3:
                    st.metric(
                        "Non-Toxic",
                        f"{ensemble_probs[2]:.3f}",
                        delta=f"{ensemble_probs[2] - 0.333:.3f}"
                    )
                
                # Model Results
                if show_model_details:
                    st.subheader("🔍 HateBERT Model Results")
                    probs = model_probs["hatebert"]
                    cols = st.columns(3)
                    with cols[0]:
                        st.write(f"Severe: {probs[0]:.3f}")
                    with cols[1]:
                        st.write(f"Moderate: {probs[1]:.3f}")
                    with cols[2]:
                        st.write(f"Non-Toxic: {probs[2]:.3f}")
                
                # Export results
                st.subheader("💾 Export Results")
                results_dict = {
                    "text": user_input,
                    "prediction": prediction,
                    "timestamp": datetime.now().isoformat(),
                    "hatebert_probabilities": {
                        "severely_toxic": float(ensemble_probs[0]),
                        "moderately_toxic": float(ensemble_probs[1]),
                        "non_toxic": float(ensemble_probs[2])
                    },
                    "model_used": "hatebert"
                }
                
                st.download_button(
                    "📥 Download Results (JSON)",
                    data=str(results_dict),
                    file_name=f"toxicity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
