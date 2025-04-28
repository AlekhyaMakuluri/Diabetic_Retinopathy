# Diabetic Retinopathy Detection System

This project implements an advanced diabetic retinopathy detection system using Graph Neural Networks (GNN), Swin Transformers, and Self-Supervised Learning (SSL) on OCT and fundus images.

## Features

- Multi-modal analysis using both OCT and fundus images
- Advanced deep learning models including GNN and Swin Transformers
- Self-supervised learning for improved feature extraction
- React-based web interface for easy interaction
- Real-time inference capabilities

## Project Structure

```
.
├── backend/
│   ├── models/         # Neural network model implementations
│   ├── data/          # Data handling and preprocessing
│   └── utils/         # Utility functions and helpers
└── frontend/
    ├── src/           # React source code
    ├── public/        # Static files
    └── components/    # React components
```

## Setup Instructions

### Backend Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the datasets:
   - OCT images: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
   - Fundus images: https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered

4. Place the datasets in the `backend/data` directory

### Frontend Setup

1. Install Node.js dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

## Usage

1. Start the backend server:
   ```bash
   python backend/app.py
   ```

2. Access the web interface at `http://localhost:3000`

## Model Architecture

The system uses a multi-modal approach combining:
- Graph Neural Networks for structural analysis
- Swin Transformers for hierarchical feature extraction
- Self-Supervised Learning for improved representation learning

## License

MIT License 