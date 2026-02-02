# 🔥 Wildfire Risk Prediction

A machine learning project for predicting wildfire risk using environmental parameters, meteorological data, and deep learning models (CNN-LSTM architecture).

[![License](https://img.shields.io/badge/License-Other-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)](https://www.tensorflow.org/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Model Architecture](#model-architecture)
- [HPC/SLURM Support](#hpcslurm-support)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project aims to assess wildfire risk using machine learning techniques by analyzing various environmental and meteorological parameters. The system leverages data from Copernicus ERA5, processes spatial-temporal features, and trains deep learning models to predict fire occurrence days.

## ✨ Features

- **Multi-source Data Integration**: Combines Copernicus ERA5 meteorological data with fire occurrence records
- **Advanced Feature Engineering**: Processes 37+ environmental and temporal features
- **CNN-LSTM Architecture**: Utilizes hybrid deep learning models for spatial-temporal pattern recognition
- **Imbalanced Data Handling**: Implements techniques to handle fire vs. non-fire day imbalance
- **HPC Cluster Support**: SLURM job scripts for running on high-performance computing environments
- **Comprehensive Pipeline**: End-to-end workflow from data collection to model evaluation

## 📁 Project Structure

```
Wildfire_risk_prediction/
│
├── scripts/                      # Python scripts for data processing and modeling
│   ├── data_collection/          # Scripts for data collection from various sources
│   ├── data_processing/          # Scripts for data cleaning, merging, and preprocessing
│   ├── cnn_lstm/                 # CNN-LSTM model implementation
│   ├── modeling/                 # Model training and evaluation scripts
│   └── utils/                    # Helper functions and utilities
│
├── notebooks/                    # Jupyter notebooks for exploration and analysis
│
├── slurm_jobs/                   # SLURM job scripts for HPC cluster execution
│
├── reports/                      # Generated reports, figures, and logs
│   ├── figures/                  # Visualizations and plots
│   └── logs/                     # Experiment logs and job outputs
│
├── tests/                        # Unit and integration tests
│
├── raw_data_wout_cffdrs/         # Raw data storage
├── parked_data/                  # Archived datasets
├── archive/                      # Historical versions and backups
│
├── requirements.txt              # Python package dependencies
├── cedar_requirements.txt        # Cedar HPC-specific requirements
├── training_parameters.json      # Model hyperparameters
├── settings.json                 # Project configuration
├── process.sh                    # Data processing pipeline script
├── train_nn.sh                   # Neural network training script
├── install_dependencies.sh       # Dependency installation script
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Access to HPC cluster with SLURM for large-scale training

### Local Setup

1. **Clone the repository:**

```bash
git clone https://github.com/Ibzo1/Wildfire_risk_prediction.git
cd Wildfire_risk_prediction
```

2. **Create and activate a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

Or use the installation script:

```bash
bash install_dependencies.sh
```

### HPC Setup (Cedar)

For running on Cedar or other HPC clusters:

```bash
pip install -r cedar_requirements.txt
```

## 💻 Usage

### 1. Data Collection

Collect meteorological and fire occurrence data:

```bash
python scripts/data_collection/collect_data.py
```

### 2. Data Processing

Process and prepare data for modeling:

```bash
bash process.sh
```

### 3. Model Training

Train the CNN-LSTM model:

```bash
bash train_nn.sh
```

Or submit a SLURM job on HPC:

```bash
sbatch slurm_jobs/train_model_job.sbatch
```

### 4. Model Evaluation

Evaluate model performance and generate visualizations:

```bash
python model_output_visualizer_prep.py
```

## 📊 Data Sources

- **Copernicus ERA5**: Meteorological reanalysis data
- **Fire Occurrence Data**: Historical wildfire records
- **OpenStreetMap**: Geographic and land use data (cached in `osm_cache/`)
- **Additional environmental parameters**: Vegetation indices, topography, etc.

## 🧠 Model Architecture

The project implements a **CNN-LSTM hybrid architecture**:

- **CNN layers**: Extract spatial features from gridded environmental data
- **LSTM layers**: Capture temporal dependencies and sequential patterns
- **Feature count**: 37 input features
- **Training parameters** (configurable in `training_parameters.json`):
  - Batch size: 10
  - Epochs: 30
  - Learning rate: 0.003

### Key Technologies

- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Model evaluation and preprocessing
- **imbalanced-learn**: Handling class imbalance
- **xarray/netCDF4**: Multi-dimensional data processing
- **Cartopy/GeoPandas**: Geospatial analysis and visualization

## 🖥️ HPC/SLURM Support

The project includes SLURM job scripts for running on high-performance computing clusters:

- `collect_data_job.sbatch`: Data collection job
- `preprocess_data_job.sbatch`: Data preprocessing job
- `train_model_job.sbatch`: Model training job
- `evaluate_model_job.sbatch`: Model evaluation job

Submit jobs using:

```bash
sbatch slurm_jobs/<job_script>.sbatch
```

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

## 📈 Results

Results and visualizations are saved in the `reports/` directory:

- Model performance metrics
- Feature importance plots
- Prediction visualizations
- Training logs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.


## 🙏 Acknowledgments

- Copernicus Climate Change Service for ERA5 data
- ECMWF for climate data access tools
- Contributors and maintainers of open-source libraries used in this project

## 📞 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This project is under active development. Features and documentation may change.

