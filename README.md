# 🏢 DevelopersHub Corporation  
### AI/ML Engineering – Advanced Internship Tasks  

---

# 🏠 Multimodal Housing Price Prediction (Images + Tabular Data)

👩‍💻 **Developer:** Aafia Azhar 

---

## 📌 Description
This project implements a **multimodal machine learning model** to predict housing prices using both:
- **Structured tabular data** (e.g., number of rooms, square footage, location, etc.)
- **Unstructured image data** (house photos)

By combining **CNN-based image features** with **tabular features**, the model provides more robust and accurate price predictions compared to using either modality alone.

---

## 🎯 Objectives
- Predict housing prices using both tabular and image data  
- Extract deep features from images using **Convolutional Neural Networks (CNNs)**  
- Perform feature fusion between image embeddings and tabular features  
- Train a regression model on the fused features  
- Evaluate model performance using **MAE** and **RMSE**  

---

## 🛠️ Tech Stack & Libraries
- **Python**
- **Pandas / NumPy** – data preprocessing  
- **Matplotlib / Seaborn** – visualization  
- **Scikit-learn** – regression metrics (MAE, RMSE)  
- **PyTorch** (or TensorFlow/Keras) – CNNs + MLP fusion model  

---


## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/aafia1/housing-multimodal.git
   cd housing-multimodal
2. Install dependencies:
pip install -r requirements.txt

3. Prepare dataset:
Place tabular CSV in data/houses.csv
Place house images in data/images/

4. Run the Jupyter Notebook:
jupyter notebook notebooks/multimodal_housing.ipynb

---

## 📊 Evaluation

The model is evaluated using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

These metrics help compare the performance of:

Tabular-only model

Image-only model

Multimodal fusion model

---

## 🎓 Skills Gained

Multimodal machine learning

CNN-based feature extraction

Feature fusion (image + tabular)

Regression modeling & evaluation


---

## ✅ Results

Tabular-only baseline performance
Image-only baseline performance
Multimodal model outperforms single-modality models with lower MAE & RMSE

---

## 📌 Future Work

Use transfer learning with pre-trained CNNs (ResNet, EfficientNet)
Experiment with attention-based fusion for better feature combination
Deploy as a Streamlit web app for real-time house price predictions
