# Noise Monitoring Dashboard (K-Means)

A Streamlit dashboard and Jupyter Notebook for exploring microphone-derived acoustic measurements and training a K-Means clustering model.

## Project contents

```text
noise_monitoring_github/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── noise_data.csv
├── models/
│   └── kmeans_noise_model.joblib
└── notebooks/
    └── Noise_Monitoring_KMeans_Training.ipynb
```

## Run locally

Python 3.11 or 3.12 is recommended. From this folder:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit.

## Train or retrain the model in Jupyter

```bash
pip install -r requirements.txt
pip install notebook matplotlib
jupyter notebook
```

Open `notebooks/Noise_Monitoring_KMeans_Training.ipynb` and run the cells from top to bottom. The notebook explains the feature columns, scaling, K-Means, cluster naming, evaluation limitations, and saves the model bundle to `models/kmeans_noise_model.joblib`.

If you retrain the model, use the model produced by the notebook together with the updated labels in the CSV. The model is unsupervised, so standard classification accuracy is not appropriate without independent ground-truth labels.

## Deploy publicly with GitHub + Streamlit Community Cloud

1. Create a **public** repository on GitHub, for example `noise-monitoring-dashboard`.
2. Upload the contents of this project folder (the files inside the folder, not the ZIP itself).
3. Check that `app.py`, `requirements.txt`, `data/noise_data.csv`, and `models/kmeans_noise_model.joblib` appear in the repository.
4. Sign in to Streamlit Community Cloud with GitHub and choose **Create app**.
5. Select your repository, branch (usually `main`), and main file path `app.py`.
6. Deploy. Streamlit installs packages from `requirements.txt`.
7. Share the generated public app URL.

The GitHub repository being public makes committed data and model files public too. Review the CSV for private or identifying information before publishing.

## Important limitations

- `estimated_level_dbfs` is a digital full-scale level, **not calibrated acoustic dB SPL**.
- The labels Quiet / Moderate / Noisy are assigned by sorting the learned clusters by average dBFS. They are relative categories, not regulatory or health thresholds.
- The dashboard uses stored measurements and offers manual feature-based predictions. It does not capture live microphone audio in Streamlit Cloud.
- The original command-line recorder uses `sounddevice`; it is not required by the hosted dashboard.
- `joblib` model files should only be loaded from sources you trust.

## Troubleshooting

- **Model file error:** run the notebook and confirm the model exists in `models/`.
- **Missing package:** check `requirements.txt` and redeploy.
- **Streamlit app not starting:** review the app logs in Streamlit Community Cloud.
- **Model version mismatch:** retrain with the dependencies listed in `requirements.txt` and commit the new model bundle.
