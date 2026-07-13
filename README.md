# 🏥 Healthcare Fraud Detection System

A machine learning-powered web application for detecting fraudulent healthcare claims using XGBoost classification.

## Live Demo

- **Streamlit App:** https://healthcare-fraud-detection-ykangcysdfeaj4tg942yqk.streamlit.app/

## Features

✨ **Key Capabilities:**
- **Single Provider Prediction**: Analyze individual healthcare provider claims for fraud risk
- **Batch CSV Prediction**: Process multiple providers at once through CSV upload
- **Real-time Risk Assessment**: Instant fraud probability scoring
- **Interactive Dashboard**: User-friendly Streamlit interface

## Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 95.21% |
| **ROC-AUC** | 98.94% |
| **F1-Score** | 95.14% |

## Tech Stack

- **Framework**: Streamlit
- **ML Model**: XGBoost
- **Data Processing**: Pandas, NumPy
- **Scaling**: Scikit-learn StandardScaler

## Features Used

The model analyzes 18 key healthcare provider features:
- Total Claims and Reimbursement metrics
- Patient demographics (Avg Age, Unique Patients)
- Chronic disease patterns (Alzheimer's, Heart Failure, Kidney Disease, Cancer, Diabetes, Depression, Stroke)
- Inpatient claim statistics
- Unique physician count

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fraud_detection_app.git
cd fraud_detection_app
```

2. **Create a virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Single Prediction
1. Navigate to "🔍 Single Provider Prediction"
2. Enter healthcare provider metrics
3. Click "Predict Fraud Risk"
4. View fraud probability and risk classification

### Batch Prediction
1. Navigate to "📂 Batch CSV Prediction"
2. Upload a CSV file with provider data
3. Results will be displayed and downloadable

## CSV Format

For batch predictions, ensure your CSV contains these columns:
```
Total_Claims,Total_Reimbursed,Avg_Reimbursed,Max_Reimbursed,
Total_Deductible,Avg_Deductible,Unique_Patients,Inpatient_Claims,
Unique_Physicians,Avg_Age,Avg_InscClaimAmtReimbursed,
Total_Alzheimer,Total_Heartfailure,Total_KidneyDisease,
Total_Cancer,Total_Diabetes,Total_Depression,Total_stroke
```

## Project Structure

```
fraud_detection_app/
├── app.py                    # Main Streamlit application
├── xgboost_fraud_model.pkl   # Trained XGBoost model
├── scaler.pkl                # Feature scaler
├── feature_columns.json      # Feature names configuration
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore               # Git ignore rules
```

## Model Details

- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Training Data**: Historical healthcare provider claims
- **Output**: Fraud probability (0-1 scale)
- **Decision Threshold**: 0.5

## Files

- `app.py` - Main application code with Streamlit UI
- `xgboost_fraud_model.pkl` - Serialized trained model
- `scaler.pkl` - Feature scaling transformer
- `feature_columns.json` - Expected feature names for model input

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

⚠️ **Important**: This system is designed as a decision support tool. All flagged cases should be reviewed by qualified fraud investigation professionals. The model's predictions should not be used as the sole basis for fraud determination.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Built with ❤️ for healthcare fraud detection**
