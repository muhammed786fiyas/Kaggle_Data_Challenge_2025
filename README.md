## 📌 **Author**

* Name: **MUHAMMED FIYAS**  
* Roll no.: **DA25M018**  
* *M.Tech DS&AI, IIT Madras*

---

# 📊 **DA5401 – 2025 Data Challenge: Fitness Score Prediction for Conversational AI Metrics**

This project focuses on building a robust prediction model for the **metric–prompt–response fitness score (0–10)** used in AI evaluation systems.  
The problem involves modelling **semantic alignment** between an evaluation metric and a generated response, using **contrastive embeddings**, **interaction features**, and a **dual-MLP ensemble** with calibration.

This implementation achieved a **Public RMSE of 3.108**, significantly outperforming classical ML baselines (~4.6 RMSE).

---

## 🎯 **Objective**

The goal of this challenge is to accurately predict the **fitness score** assigned by an evaluation system for a given:

- **Metric name**
- **System prompt**
- **User prompt**
- **Assistant response**

We aim to:

1. Build embeddings that capture semantic relationships.  
2. Construct feature engineering layers that quantify similarity/mismatch.  
3. Generate contrastive negative samples to improve robustness.  
4. Train a dual-MLP ensemble for non-linear regression.  
5. Apply calibration to align predictions with discrete scoring patterns.

> 💡 *Note:* The problem involves highly imbalanced targets (most scores are 9–10), making calibration and augmentation essential.

---

## 📂 **Project Structure**
```
DA5401_Data_Challenge_2025/
├── kaggle_data_challenge_2025.ipynb # Full notebook (EDA → Model → Results)
├── Kaggle_Data_Challenge_2025_report.pdf # Full project report
└── README.md
```
---


## 🛠️ **What's Included**

| Section | Description |
| :-- | :-- |
| **Part A: Dataset & Preprocessing** | Loads and formats the metric, prompt, and response fields. Creates structured combined text. Handles missing system prompts and empty responses. |
| **Part B: Embedding Generation** | Uses `multilingual-e5-base` to encode metric names and combined text. Produces dense 1024-d embeddings for each. |
| **Part C: Contrastive Negative Sampling** | Generates mismatched examples using three strategies: shuffled responses, noise injection, and random metric swaps. Expands dataset to 20,000 samples. |
| **Part D: Feature Engineering** | Constructs a 4097-dimensional feature vector using concatenation, absolute difference, Hadamard product, cosine similarity, and L2 distance. |
| **Part E: Dual-MLP Modeling** | Two neural models trained independently with SmoothL1 loss; predictions averaged. 5-fold cross validation used to stabilize results. |
| **Part F: Calibration** | Uses Isotonic Regression to correct prediction bias caused by target skewness. |
| **Part G: Final Submission** | Averages predictions from all folds and both models, applies calibration, clips to [0–10], and rounds for submission. |

---

## 📊 **EDA Highlights**

- **Score distribution:**  
  - Highly skewed; ~80% of samples are in **[8–10]**  
- **Text lengths:**  
  - User prompt avg length: **43 words**  
  - Response avg length: **132 words**  
- **Missing values:**  
  - `system_prompt` missing in **1549** training rows  
- **Most common metric families:**  
  - Under-rejection  
  - Out-of-scope  
  - Jailbreak prompts  
  - Instruction misuse  

> Figures for all EDA components are in the main report.

---

## 🧠 **Modeling Approach**

### **1. Embeddings**
- Model: **intfloat/multilingual-e5-base**
- Output dim: **1024**
- Two embeddings generated:
  - Metric embedding
  - Combined text embedding  
- Embeddings fused using engineered interaction features

---

### **2. Contrastive Negative Samples**

Created **three categories** of negatives:
1. **Shuffled responses** → metric kept constant  
2. **Gaussian noise injection** → degrade semantic signal  
3. **Metric–text swaps** → mismatched content  

Purpose:
- Improve discrimination  
- Reduce overfitting  
- Increase data variety  

Final training set size: **20,000** samples.

---

### **3. Feature Engineering**

The final feature vector (size **4097**) includes:

- 1024-d metric embedding  
- 1024-d text embedding  
- 1024-d absolute difference  
- 1024-d element-wise product  
- 1-d cosine similarity  
- 1-d Euclidean distance  
- 1-d optional text-length features  

These features strengthen semantic comparison between metric and text.

---

### **4. Neural Network Models**

Two MLP models used:

#### **Model A**
- Layers: 4097 → 1024 → 512 → 128 → 1  
- Activation: **GELU**  
- Regularization: LayerNorm + Dropout  
- Strength: Captures complex non-linear patterns  

#### **Model B**
- Layers: 4097 → 768 → 128 → 1  
- Activation: **ReLU**  
- Strength: Smoother, low-variance predictions  

**Optimiser:** AdamW  
**Loss:** SmoothL1Loss  
**Training:** 5-fold CV, 20 epochs per fold

---

## 📈 **Results**

### **Per-fold Validation RMSE:**
| Fold | RMSE |
| :-- | :--: |
| Fold 0 | 2.7911 |
| Fold 1 | 2.7329 |
| Fold 2 | 2.7508 |
| Fold 3 | 2.6579 |
| Fold 4 | 2.7135 |
| **Mean OOF** | **2.7296** |

### **Calibration**
- Linear isotonic calibration applied  
- OOF RMSE improved from **2.7296 → 2.7258**

### **Public Leaderboard**
- **Final RMSE:** ⭐ **3.108**  

A strong improvement over classical ML baselines (XGBoost ≈ 4.6 RMSE).

---

## 🧩 **Key Insights**

- Multilingual embeddings help handle diverse linguistic prompts.  
- Interaction features provide essential relational information.  
- Negative samples dramatically reduce overfitting.  
- Ensemble of two MLPs achieves better stability than a single model.  
- Calibration is crucial for discrete score alignment.  

---

## 🔧 **Technologies Used**
- Python  
- PyTorch (MLP implementation)  
- Sentence-Transformers (E5 embeddings)  
- Scikit-Learn (calibration, preprocessing)  
- NumPy, Pandas  
- Google Colab GPU (Tesla T4)

---

## 🎯 **Conclusion**

The combination of:
- Strong multilingual embeddings  
- Carefully engineered negative samples  
- Rich interaction features  
- Dual-MLP ensemble  
- Calibration  

produced a competitive and robust solution, achieving **3.108 RMSE** on the public leaderboard.

For full details, refer to **Kaggle_Data_Challenge_2025_report.pdf**.

