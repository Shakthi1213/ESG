# ESG Score & Risk Prediction System — Deployment Guide

## 📁 Required Files (all in the same folder)

```
your-project/
├── app.py
├── requirements.txt
├── esg_score_model.pkl
├── esg_risk_model_modelC.pkl
├── scaler__1_.pkl
├── esg_score_features.pkl
└── esg_risk_features_modelC.pkl
```

---

## 🚀 Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial ESG app"
gh repo create esg-prediction-app --public --source=. --push
```
Or manually create a repo on github.com and upload the files.

### Step 2 — Connect to Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`
5. Click **Deploy** — your app will be live in ~60 seconds

> **Important:** `requirements.txt` pins `scikit-learn==1.6.1` to match the version used to train and pickle the models. Do **not** change this version.
>
> **Important:** all five `.pkl` files listed above must be committed to the repository. If any are missing, the app will stop at startup and show which artifact is missing.

---

## 💻 Run Locally

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

App will open at **http://localhost:8501**

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `_pickle.UnpicklingError` | Ensure scikit-learn==1.6.1 is installed (matches training version) |
| `Feature mismatch` | The app auto-aligns features from `.pkl` files — no action needed |
| Models not found | Place all `.pkl` files in the **same directory** as `app.py` |
| Port in use | Run `streamlit run app.py --server.port 8502` |

---

## 🧠 App Architecture

```
User Input (Sidebar)
      │
      ▼
Step 1: Build ESG Score Input (8 features from esg_score_features.pkl)
      │
      ▼
Step 2: Scale with scaler__1_.pkl → Predict with esg_score_model.pkl
      │
      ▼ Predicted ESG Score
      │
Step 3: Build Risk Input (OHE features from esg_risk_features_modelC.pkl)
        — injects predicted ESG score as a feature
      │
      ▼
Step 4: Classify with esg_risk_model_modelC.pkl → High / Medium / Low
      │
      ▼
Step 5: Display Score + Risk Level + Tailored Recommendations
```
