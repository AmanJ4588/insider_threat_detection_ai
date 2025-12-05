# Insider Threat Detection System

An AI-powered system for detecting insider threats using machine learning. This project combines **unsupervised anomaly detection** (Isolation Forest) with **supervised classification** (XGBoost) for comprehensive threat identification and explainability.

## 🎯 Overview

This system analyzes user behavior within an organization to identify potential insider threats. It leverages:

- **Hybrid ML Approach**: Combines unsupervised anomaly detection with supervised classification
- **Explainability**: XAI integration for transparent threat explanations
- **Real-time Inference**: FastAPI-based REST API for live threat analysis
- **Production Ready**: Deployment-ready with containerization support

## 📁 Project Structure

```
insider_threat_detection_system/
├── data_transformation_and_preprocessing/    # ETL Pipeline
│   ├── 01_ingest_and_compress/              # CSV to Parquet conversion
│   ├── 02_load_to_duckdb/                   # Data loading into DuckDB
│   ├── 03_aggregate_features/               # Feature aggregation by user
│   ├── 04_unsupervised_data_preprocessing/  # Unsupervised data prep & EDA
│   └── 05_supervised_data_preprocessing/    # Supervised data prep & EDA
├── model_training_and_evaluation/           # Model training notebooks
│   ├── supervised_model_XGBClassifier/      # XGBoost training & evaluation
│   └── unsupervised_model_isolation_forest/ # Isolation Forest training
├── inference/                                # Inference pipelines
│   ├── 01_fake_data_generation/             # Test data generation
│   ├── 02_inference_hybrid_approach/        # Combined inference pipeline
│   └── 03_XAI_Integration/                  # Explainability integration
├── deployment/                              # Production deployment
│   ├── server.py                            # FastAPI application
│   ├── inference_with_XAI_package.py        # Unified inference engine
│   └── requirements.txt                     # Deployment dependencies
├── requirements.txt                         # Core dependencies
└── .gitignore                               # Git ignore patterns
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/AmanJ4588/insider_threat_detection_ai
   cd insider_threat_detection_system
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the CERT R4.2 Dataset**
   - Download from [kilthub.cmu.edu](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247)
   - Place the dataset in the `dataset/` directory following the existing structure

### Running the Deployment Server

Start the FastAPI server for real-time inference:

```bash
cd deployment
python server.py
```

The API will be available at `http://localhost:8000`

**API Endpoints:**

- `GET /` - Health check
- `POST /analyze` - Analyze user data for threats

**Example Request:**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "feature_1": 0.5,
    "feature_2": 0.8,
    ...
  }'
```

**Response:**

```json
{
  "user_id": "user123",
  "risk_score": 75,
  "anomaly_score": 82,
  "prediction": "insider_threat",
  "confidence": 0.92,
  "explanation": "..."
}
```

## 🔄 Data Pipeline

### 1. **Data Ingestion & Compression** (`01_ingest_and_compress/`)

- Converts raw CSV files to Parquet format for efficiency
- Reduces storage footprint while maintaining data integrity

### 2. **Loading into DuckDB** (`02_load_to_duckdb/`)

- Loads Parquet files into DuckDB for fast querying
- Creates structured database tables for analysis
- Scripts: `parquet_to_duckDB.py`, `answers_to_duckDB.py`

### 3. **Feature Aggregation** (`03_aggregate_features/`)

- Aggregates events by user to create behavioral features
- Generates datasets for both supervised and unsupervised learning
- Notebooks: `tables_aggregation_by_user.ipynb`, `unsupervised_tables_export.ipynb`

### 4. **Unsupervised Data Preprocessing** (`04_unsupervised_data_preprocessing/`)

- EDA on unlabeled user data
- Feature scaling and normalization
- Outlier detection and handling

### 5. **Supervised Data Preprocessing** (`05_supervised_data_preprocessing/`)

- Labels insider threat data using ground truth
- EDA on labeled dataset
- Handles class imbalance

## 🤖 Machine Learning Models

### Unsupervised Model: Isolation Forest

- **Purpose**: Detect anomalous user behavior patterns
- **Training**: `model_training_and_evaluation/unsupervised_model_isolation_forest/`
- **Output**: Anomaly scores (0-100) indicating deviation from normal behavior
- **Advantages**: Doesn't require labeled data, detects novel attack patterns

### Supervised Model: XGBoost Classifier

- **Purpose**: Classify users as insider threats based on labeled examples
- **Training**: `model_training_and_evaluation/supervised_model_XGBClassifier/`
- **Components**:
  - `hyperparameter_tuning.ipynb` - Hyperparameter optimization
  - `model_training_and_evaluation.ipynb` - Training and evaluation
  - `best_threshold.ipynb` - Optimal decision threshold selection
- **Output**: Risk scores (0-100) and threat classification

### Hybrid Approach

- Combines both models for comprehensive threat detection
- Unsupervised scores identify behavioral anomalies
- Supervised scores provide labeled threat classification
- Final decision based on both signals

## 📊 Inference Pipeline

### Hybrid Inference (`inference/02_inference_hybrid_approach/`)

```
User Data
    ↓
[Unsupervised] → Isolation Forest → Anomaly Score
    ↓
[Feature Processing]
    ↓
[Supervised] → XGBoost Classifier → Risk Score
    ↓
[XAI Integration]
    ↓
Threat Assessment + Explanation
```

### XAI Integration (`inference/03_XAI_Integration/`)

- SHAP values for model explainability
- Feature importance analysis
- Transparent threat explanations

## 📦 Key Dependencies

### Core ML Libraries

- **scikit-learn** (1.7.2) - Machine learning algorithms
- **xgboost** (3.1.1) - Gradient boosting classifier
- **pandas** (2.3.3) - Data manipulation
- **numpy** (2.3.3) - Numerical computing
- **tensorflow** (2.20.0) - Deep learning framework

### Data Processing

- **duckdb** (1.4.0) - Fast SQL database
- **pyarrow** (17.0.0) - Columnar data format
- **fastavro** (1.11.1) - Avro serialization

### Deployment & API

- **fastapi** (0.119.0) - REST API framework
- **uvicorn** (0.34.0) - ASGI server
- **gunicorn** (23.0.0) - WSGI HTTP server

### Explainability

- **shap** (0.50.0) - SHAP values for interpretability

### Monitoring & Utilities

- **tensorboard** (2.20.0) - Training visualization
- **jupyter** (1.1.1) - Interactive notebooks
- **pytest** - Testing framework

See `requirements.txt` for full dependency list.

## 🧪 Testing & Validation

### Fake Data Generation

Generate test data for inference and validation:

```bash
python inference/01_fake_data_generation/fake_data_generator.py
```

### Running Inference

Execute the hybrid inference pipeline:

```bash
python inference/02_inference_hybrid_approach/hybrid_inference.py
```

## 🔐 Production Deployment

### Using Docker (Recommended)

```bash
docker build -t insider-threat-detection:latest .
docker run -p 8000:8000 insider-threat-detection:latest
```

### Manual Deployment

```bash
cd deployment
pip install -r requirements.txt
gunicorn server:app --bind 0.0.0.0:8000 --workers 4
```

## 📈 Model Performance

Models are evaluated on:

- **Precision** - Minimize false positives
- **Recall** - Catch actual threats
- **F1-Score** - Balance precision and recall
- **ROC-AUC** - Overall discrimination ability

See evaluation notebooks in `model_training_and_evaluation/` for detailed metrics.

## 🔍 Dataset Information

This project uses the **CERT R4.2 Insider Threat Dataset**:

- Realistic synthetic user behavior data
- Multiple insider threat scenarios
- Ground truth labels for evaluation
- Includes file access, email, and web browsing events

**Note**: The dataset is not included in this repository. Download from [kilthub.cmu.edu](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247) and place in the `dataset/` directory.

## 📝 Configuration

Key configuration files:

- `requirements.txt` - Python dependencies
- `deployment/requirements.txt` - Deployment-specific dependencies
- Model paths defined in inference scripts

## 🛠️ Development

### Running Notebooks

All analysis and training uses Jupyter notebooks:

```bash
jupyter notebook
```

Navigate to:

- Data preprocessing: `data_transformation_and_preprocessing/`
- Model training: `model_training_and_evaluation/`

### Adding New Features

1. Update preprocessing pipeline in `data_transformation_and_preprocessing/`
2. Retrain models in `model_training_and_evaluation/`
3. Update inference pipeline in `inference/`

## 📚 Additional Resources

- CERT Insider Threat Dataset: [kilthub.cmu.edu](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247)
- SHAP Documentation: [github.com/slundberg/shap](https://github.com/slundberg/shap)
- XGBoost Documentation: [xgboost.readthedocs.io](https://xgboost.readthedocs.io)

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and validation
4. Submit a pull request

## 📄 License

Check `dataset/answers/license.txt` for dataset licensing information.

## ⚠️ Disclaimer

This system is designed for research and organizational security purposes. Results should be validated by security experts before taking any action against users.

---

**Last Updated**: December 2025

For questions or issues, please contact the development team or open an issue on the repository.
