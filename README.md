<div align="center">

# 🏔️ Natural Landscape Classifier
### *PyTorch Scene Recognition Web App (Custom CNN vs. ResNet-18 Transfer Learning)*

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://landscape-classifier-app.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

**An interactive computer vision web application trained on the Intel Image Classification dataset to classify natural and urban scenes across 6 categories (`Buildings`, `Forest`, `Glacier`, `Mountain`, `Sea`, `Street`). Users can toggle live between a custom-built CNN and a fine-tuned ResNet-18 transfer learning pipeline.**

</div>

---

## 🧠 Model Architectures Compared

| Model Option | Input Resolution | Architecture Details | Key Takeaway |
| :--- | :---: | :--- | :--- |
| **1. ResNet-18 (Transfer Learning)** | `224 × 224` | Pretrained `ResNet-18` backbone with a custom 3-layer classifier head (`512 → 256 → 128 → 6`) and `0.3` Dropout. | Higher generalization accuracy and stronger feature extraction on complex textures. |
| **2. Custom CNN (From Scratch)** | `150 × 150` | 2 convolutional blocks (`Conv2d → ReLU → BatchNorm2d → MaxPool2d`) with 64 hidden channels followed by a dense classifier. | Lightweight baseline demonstrating spatial feature learning from scratch without pretrained weights. |

---

## 📂 Repository Structure

```text
landscape-classifier-app/
├── notebooks/
│   ├── 01_custom_cnn_sandbox.ipynb              # Custom CNN architecture & training loop
│   └── 02_resnet18_transfer_learning.ipynb      # ResNet-18 transfer learning & model comparison
├── app.py                                       # Streamlit frontend UI & caching
├── model.py                                     # PyTorch model definitions & GitHub Release weight loaders
├── utils.py                                     # Image preprocessing transforms & softmax inference
├── requirements.txt                             # Production dependencies
└── README.md
```

---

## 💻 Local Quick Start

```bash
# 1. Clone the repository
git clone [https://github.com/vikassinngh123/landscape-classifier-app.git](https://github.com/vikassinngh123/landscape-classifier-app.git)
cd landscape-classifier-app

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 3. Install dependencies and launch the Streamlit app
pip install -r requirements.txt
streamlit run app.py
```
*(Model weights `custom_cnn_model.pth` and `resnet18_model.pth` are automatically downloaded from GitHub Releases on first launch).*

---

<div align="center">
<b>Built by <a href="https://github.com/vikassinngh123">Vikas Singh</a></b>
</div>
