# 🧬 Roundworm Classification with Transfer Learning

This project classifies microscope images of roundworms as either **Alive** or **Dead** using Transfer Learning with MobileNetV2 and other machine learning models.

## 📂 Dataset
- 93 microscope images in `.tif` format
- Images are labeled based on filename prefixes:
  - `wormA*` = Alive
  - `wormB*`, `wormC*`, `wormD*` = Dead

## 📊 Models Compared
| Model            | Accuracy | F1-Score | Notes                   |
|------------------|----------|----------|--------------------------|
| CNN (base)       | 68%      | 0.60     | Overfit, low recall for Alive |
| CNN + Class Weights | 53%   | 0.55     | Balanced class weights     |
| CNN + Augmentation | 58%   | 0.58     | Minor improvement         |
| Fine-tuned MobileNetV2 | 58% | 0.58   | Overfitting persists      |
| Random Forest + SMOTE | **96%** | **0.96** | Best performer           |
| SVM + SMOTE      | **96%**  | **0.96** | Equally strong            |

## 🧪 Techniques Used
- Transfer Learning (MobileNetV2)
- Data Augmentation
- Class Balancing (SMOTE)
- Evaluation: Confusion Matrix, Classification Report
- Feature-Based Models (Random Forest, SVM)

## 📁 Files
- `roundworm_classification.ipynb`: Main Colab notebook
- `WormImages/`: Dataset folder
- `README.md`: Project documentation

## 🚀 Future Work
- Deploy on Streamlit for interactive classification
- Expand dataset with more diverse worm states
- Apply focal loss or contrastive learning

---

## 💻 Setup

Install dependencies with:

```bash
pip install -r requirements.txt
