import streamlit as st
import joblib
import re
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import altair as alt

# ============================================================
# 🔹 Page Configuration
# ============================================================
st.set_page_config(
    page_title="NLP Job Role Classifier",
    page_icon="🧠",
    layout="wide",
)

# ============================================================
# 🔹 Custom CSS for Enhanced UI
# ============================================================
st.markdown("""
    <style>
    /* Background gradient */
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #e0e0e0;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #64b5f6, #9575cd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }

    .sub-title {
        text-align: center;
        color: #cfd8dc;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
    }

    /* Text area and buttons */
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #37474f;
        background-color: #263238;
        color: #f5f5f5 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #512da8, #1976d2);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        height: 45px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0px 0px 15px rgba(100,181,246,0.4);
    }

    /* Result card */
    .result-card {
        padding: 1.2rem;
        border-radius: 15px;
        background-color: #1c1f24;
        color: #e0e0e0;
        box-shadow: 0px 3px 15px rgba(0,0,0,0.4);
        margin-top: 1rem;
        border-left: 4px solid #64b5f6;
    }

    /* KPI cards */
    .kpi-box {
        background-color: #1e272e;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        color: #e0e0e0;
        box-shadow: 0 0 15px rgba(255,255,255,0.05);
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #82b1ff;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #b0bec5;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #424242;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 🔹 Header
# ============================================================
st.markdown('<h1 class="main-title">🧠 NLP Resume Classification App</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-powered Job Role Prediction using NLP Techniques</p>', unsafe_allow_html=True)

# ============================================================
# 🔹 Helper Functions
# ============================================================
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# ============================================================
# 🔹 Sidebar
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    dataset_choice = st.selectbox("Select Dataset", ("Dataset 1", "Dataset 2"))

if dataset_choice == "Dataset 1":
    model_path = "Dataset1.pkl"
    vectorizer_path = "tfidf_Dataset1.pkl"
    data_path = "Resume.csv"
    model_accuracies = {
        "Logistic Regression": 0.6615,
        "Naive Bayes": 0.5384,
        "SVM": 0.6923
    }
else:
    model_path = "Dataset2.pkl"
    vectorizer_path = "tfidf_Dataset2.pkl"
    data_path = "UpdatedResumeDataSet.csv"
    model_accuracies = {
        "Logistic Regression": 0.4411,
        "Naive Bayes": 0.2353,
        "SVM": 0.9411
    }

# ============================================================
# 🔹 Load Model
# ============================================================
try:
    with st.spinner("🔄 Loading model and vectorizer..."):
        time.sleep(1)
        model, tfidf = load_model_and_vectorizer(model_path, vectorizer_path)
    st.sidebar.success("✅ Model and Vectorizer Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"⚠️ Error loading model/vectorizer: {e}")
    st.stop()

# ============================================================
# 🔹 KPI Section
# ============================================================
st.markdown("### 📈 Key Performance Indicators")
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

df_preview = pd.read_csv(data_path)
best_model = max(model_accuracies, key=model_accuracies.get)
col_kpi1.markdown(f"<div class='kpi-box'><div>📂 Dataset Size</div><div class='kpi-value'>{len(df_preview):,}</div></div>", unsafe_allow_html=True)
col_kpi2.markdown(f"<div class='kpi-box'><div>🏆 Best Model</div><div class='kpi-value'>{best_model}</div></div>", unsafe_allow_html=True)
col_kpi3.markdown(f"<div class='kpi-box'><div>🎯 Best Accuracy</div><div class='kpi-value'>{model_accuracies[best_model]*100:.2f}%</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# 🔹 Input and Prediction Columns
# ============================================================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📝 Enter Job Description")
    user_input = st.text_area(
        "Paste or type a job/resume description below:",
        placeholder="Example: Seeking a data scientist with experience in NLP and Python...",
        height=180
    )
    predict_btn = st.button("🎯 Predict Category", use_container_width=True)

with col2:
    st.subheader("🔮 Prediction Results")
    result_box = st.empty()

    if predict_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a valid job description.")
        else:
            with st.spinner("✨ Analyzing text and predicting category..."):
                time.sleep(1)
                cleaned = clean_text(user_input)
                vector = tfidf.transform([cleaned])
                prediction = model.predict(vector)[0]

            result_box.markdown(f"""
                <div class='result-card'>
                    <h4>✅ Predicted Job Role / Category:</h4>
                    <h3 style='color:#82b1ff'>{prediction}</h3>
                </div>
            """, unsafe_allow_html=True)

# ============================================================
# 🔹 Model Accuracy Visualization
# ============================================================
st.markdown("---")
st.subheader("📊 Model Accuracy Comparison")

acc_df = pd.DataFrame({
    "Model": list(model_accuracies.keys()),
    "Accuracy": [v * 100 for v in model_accuracies.values()]
})

chart = (
    alt.Chart(acc_df)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    .encode(
        x=alt.X("Model", sort=None, axis=alt.Axis(labelColor='white', titleColor='white')),
        y=alt.Y("Accuracy", axis=alt.Axis(labelColor='white', titleColor='white')),
        color=alt.Color("Model", legend=None, scale=alt.Scale(scheme="blues")),
        tooltip=["Model", "Accuracy"]
    )
    .configure_view(strokeWidth=0)
    .properties(height=300, background="#1c1f24")
)
st.altair_chart(chart, use_container_width=True)

# ============================================================
# 🔹 Evaluate Model Button
# ============================================================
st.markdown("---")
if st.button("📈 Evaluate Current Model on Dataset", use_container_width=True):
    try:
        with st.spinner("🔍 Evaluating model performance on dataset..."):
            df = pd.read_csv(data_path)
            df = df.dropna(subset=['Resume', 'Category']).reset_index(drop=True)
            X = df['Resume'].apply(clean_text)
            y = df['Category']
            X_vec = tfidf.transform(X)
            y_pred = model.predict(X_vec)
            acc = accuracy_score(y, y_pred)
            time.sleep(1)
        st.success(f"🔹 Model evaluated on full dataset → **Accuracy: {acc*100:.2f}%**")
    except Exception as e:
        st.error(f"⚠️ Could not evaluate model: {e}")

# ============================================================
# 🔹 Footer
# ============================================================
st.markdown("""
<div class='footer'>
    <p>🚀 Developed by <b>Sucheti Kanere</b> | NLP Resume Classification Project</p>
</div>
""", unsafe_allow_html=True)