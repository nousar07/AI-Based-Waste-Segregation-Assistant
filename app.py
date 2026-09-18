import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Waste Segregation Assistant",
    page_icon="♻️",
    layout="centered"
)

# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #b7b7b7;
    margin-bottom: 20px;
}

.sdg-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 20px;
    background-color: #174d36;
    color: #ffffff;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 20px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
    margin-bottom: 20px;
}

.result-title {
    font-size: 25px;
    font-weight: 700;
}

.small-text {
    color: #b7b7b7;
    font-size: 14px;
}

.section-title {
    font-size: 27px;
    font-weight: 700;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "waste_classifier.keras"
)

# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ============================================================
# 5. CLASS NAMES
# ============================================================

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# ============================================================
# 6. WASTE INFORMATION
# ============================================================

waste_info = {

    "cardboard": {
        "icon": "📦",
        "name": "Cardboard Waste",
        "type": "♻️ Recyclable",
        "disposal": (
            "Keep cardboard clean and dry. Flatten boxes and place "
            "them in the appropriate recyclable paper/cardboard stream."
        ),
        "tip": (
            "Reuse cardboard boxes whenever possible before recycling them."
        )
    },

    "glass": {
        "icon": "🍾",
        "name": "Glass Waste",
        "type": "♻️ Usually Recyclable",
        "disposal": (
            "Separate glass from other waste and follow your local "
            "glass-recycling guidelines."
        ),
        "tip": (
            "Reuse suitable glass containers when safe and practical."
        )
    },

    "metal": {
        "icon": "🥫",
        "name": "Metal Waste",
        "type": "♻️ Recyclable",
        "disposal": (
            "Place clean metal cans and containers in the appropriate "
            "recyclable metal collection."
        ),
        "tip": (
            "Recycling metals helps conserve raw materials and energy."
        )
    },

    "paper": {
        "icon": "📄",
        "name": "Paper Waste",
        "type": "♻️ Recyclable",
        "disposal": (
            "Keep paper clean and dry and place it with recyclable "
            "paper waste according to local guidelines."
        ),
        "tip": (
            "Reduce paper consumption and reuse sheets whenever possible."
        )
    },

    "plastic": {
        "icon": "🧴",
        "name": "Plastic Waste",
        "type": "♻️ Depends on Plastic Type",
        "disposal": (
            "Separate recyclable plastic from general waste and follow "
            "your local recycling guidelines."
        ),
        "tip": (
            "Reduce single-use plastics and reuse suitable containers."
        )
    },

    "trash": {
        "icon": "🗑️",
        "name": "General / Other Waste",
        "type": "⚠️ Usually Non-Recyclable",
        "disposal": (
            "Place non-recyclable waste in the appropriate general-waste "
            "stream according to local rules."
        ),
        "tip": (
            "Check whether the item can be reused, repaired, or recycled "
            "before disposing of it."
        )
    }
}

# ============================================================
# 7. HEADER
# ============================================================

st.markdown(
    '<div class="sdg-badge">🌍 SDG 12 • Responsible Consumption & Production</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">♻️ AI-Based Waste Segregation Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An AI-powered tool that identifies common waste materials '
    'and provides responsible disposal guidance.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# 8. UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📷 Upload a Waste Image</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a clear image of a waste item. "
    "The AI will analyze the image and suggest an appropriate waste category."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# 9. PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image_resized = image.resize((224, 224))

    image_array = np.array(image_resized)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # IMPORTANT:
    # No MobileNetV2 preprocessing here.
    # The trained model already performs preprocessing.

    predictions = model.predict(
        image_array,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = np.argmax(probabilities)

    predicted_class = class_names[predicted_index]

    confidence = (
        float(probabilities[predicted_index]) * 100
    )

    info = waste_info[predicted_class]

    # ========================================================
    # 10. AI ANALYSIS
    # ========================================================

    st.markdown(
        '<div class="section-title">🤖 AI Analysis</div>',
        unsafe_allow_html=True
    )

    st.success(
        f"AI Prediction: {info['icon']} {info['name']}"
    )

    st.metric(
        "AI Confidence",
        f"{confidence:.2f}%"
    )

    # Confidence message

    if confidence >= 70:

        st.success(
            "✅ The AI has relatively high confidence in this prediction."
        )

    elif confidence >= 50:

        st.warning(
            "⚠️ The AI has moderate confidence. "
            "Consider verifying the waste category."
        )

    else:

        st.warning(
            "⚠️ The AI is uncertain about this image. "
            "Try uploading a clearer image."
        )

    # ========================================================
    # 11. WASTE TYPE
    # ========================================================

    st.markdown(
        '<div class="section-title">♻️ Waste Classification</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"**Category:** {info['name']}\n\n"
        f"**Classification:** {info['type']}"
    )

    # ========================================================
    # 12. TOP 3 PREDICTIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Top AI Predictions</div>',
        unsafe_allow_html=True
    )

    top_indices = np.argsort(
        probabilities
    )[-3:][::-1]

    for rank, index in enumerate(top_indices, start=1):

        category = class_names[index]

        percentage = (
            float(probabilities[index]) * 100
        )

        st.write(
            f"**{rank}. {waste_info[category]['icon']} "
            f"{category.title()} — {percentage:.2f}%**"
        )

        st.progress(
            min(max(float(probabilities[index]), 0.0), 1.0)
        )

    # ========================================================
    # 13. DISPOSAL GUIDANCE
    # ========================================================

    st.markdown(
        '<div class="section-title">🗑️ Disposal Guidance</div>',
        unsafe_allow_html=True
    )

    st.info(info["disposal"])

    # ========================================================
    # 14. SUSTAINABILITY TIP
    # ========================================================

    st.markdown(
        '<div class="section-title">🌱 Sustainability Tip</div>',
        unsafe_allow_html=True
    )

    st.success(info["tip"])

    # ========================================================
    # 15. HOW IT WORKS
    # ========================================================

    st.divider()

    with st.expander("⚙️ How This AI System Works"):

        st.write("""
        **Step 1 — Image Input**

        The user uploads an image of a waste item.

        **Step 2 — Image Processing**

        The image is resized to the format expected by the trained model.

        **Step 3 — AI Classification**

        A MobileNetV2-based deep learning model analyzes visual patterns
        in the image.

        **Step 4 — Waste Category**

        The model predicts one of six categories:

        • Cardboard  
        • Glass  
        • Metal  
        • Paper  
        • Plastic  
        • Trash

        **Step 5 — Sustainability Guidance**

        The predicted category is connected to disposal guidance and
        a sustainability tip.
        """)

    # ========================================================
    # 16. SDG 12 CONNECTION
    # ========================================================

    with st.expander("🌍 How This Project Supports SDG 12"):

        st.write("""
        **SDG 12: Responsible Consumption and Production** promotes
        sustainable management of resources and reduction of waste.

        This project supports SDG 12 by:

        • Helping users identify common waste materials.

        • Encouraging appropriate waste segregation.

        • Providing recycling and disposal guidance.

        • Promoting reuse and reduction of unnecessary waste.

        • Demonstrating how AI can support sustainability awareness.
        """)

    # ========================================================
    # 17. RESPONSIBLE AI
    # ========================================================

    with st.expander("🛡️ Responsible AI Considerations"):

        st.write("""
        **Fairness**

        The model may perform differently on waste items that are not
        well represented in its training dataset.

        **Transparency**

        The application displays the predicted category and confidence
        score rather than presenting the prediction as guaranteed.

        **Privacy**

        Users should avoid uploading images containing unnecessary
        personal or sensitive information.

        **Human Verification**

        AI predictions should be treated as decision-support information,
        especially when confidence is low.

        **Local Guidelines**

        Waste-management rules can vary by location. Users should always
        follow applicable local waste-disposal and recycling guidelines.
        """)

    # ========================================================
    # 18. LIMITATIONS
    # ========================================================

    with st.expander("⚠️ Project Limitations"):

        st.write("""
        • The model is trained on six common waste categories.

        • Performance may decrease for unusual objects or images that
          differ significantly from the training dataset.

        • Some objects can contain multiple materials, making classification
          difficult.

        • AI confidence does not guarantee that a prediction is correct.

        • Local recycling rules may differ from the general guidance
          provided by this application.
        """)

# ============================================================
# 19. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based Waste Segregation Assistant • "
    "AI for Sustainability Virtual Internship • "
    "SDG 12"
)

st.caption(
    "AI predictions are for decision support. "
    "Always follow local waste-management guidelines."
)