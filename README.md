# AI Waste Segregation Assistant 🌿♻️

An intelligent web application built with **Streamlit** and **Deep Learning (MobileNetV2)** designed to automate waste segregation. This project directly supports **SDG 12: Responsible Consumption and Production** by empowering users to classify household waste into recyclable and non-recyclable categories accurately and instantly.

---

## 🚀 Features

- **Real-Time Image Classification:** Upload an image of waste or use a sample image to classify it instantly.
- **6 Waste Categories:** Supports classification for *Cardboard, Glass, Metal, Paper, Plastic,* and *Trash*.
- **Confidence Scores:** View the prediction confidence level along with category-specific disposal instructions and recycling tips.
- **Interactive UI:** Clean, responsive, and user-friendly interface powered by Streamlit.
- **Pre-trained Deep Learning Model:** Fine-tuned MobileNetV2 architecture optimized for high accuracy and fast inference on edge devices.

---

## 🛠️ Tech Stack

- **Frontend & Web App:** Streamlit, Python
- **Deep Learning / ML:** TensorFlow / Keras (MobileNetV2)
- **Image Processing:** Pillow (PIL), NumPy

---

## 📂 Project Structure

```text
ai-waste-segregation/
│
├── app.py              # Streamlit web application
├── train_model.py      # Script for training/fine-tuning the MobileNetV2 model
├── requirements.txt    # Required Python packages
└── README.md           # Project documentation
```

---

## ⚙️ Installation & Setup

Follow these steps to set up and run the project locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-waste-segregation.git
cd ai-waste-segregation
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional / If weights are missing)
If you need to train the model from scratch or retrain on your own dataset:
```bash
python train_model.py
```

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 💡 How to Use

1. Open the local Streamlit URL provided in your terminal (usually `http://localhost:8501`).
2. Upload a clear picture of a waste item (e.g., a plastic bottle, cardboard box, or aluminum can) using the file uploader.
3. Click **Classify Waste** to see the predicted category, confidence score, and sustainable disposal recommendations.

---

## 🌍 Contributing & Future Scope
Contributions are always welcome! Feel free to open an issue or submit a pull request. 
* **Future Scope:** Integrating live camera feed detection via webcam and adding localized recycling center finders.