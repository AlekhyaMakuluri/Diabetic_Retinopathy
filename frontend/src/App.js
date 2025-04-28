import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [octImage, setOctImage] = useState(null);
  const [fundusImage, setFundusImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const handleImageUpload = (event, setImage) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please upload an image file');
        return;
      }
      setError(null);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = (e, setImage) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please upload an image file');
        return;
      }
      setError(null);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const removeImage = (setImage) => {
    setImage(null);
  };

  const handlePredict = async () => {
    if (!octImage && !fundusImage) {
      setError('Please upload at least one image');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      if (octImage) {
        const octBlob = await fetch(octImage).then(r => r.blob());
        formData.append('oct', octBlob, 'oct.jpg');
      }
      if (fundusImage) {
        const fundusBlob = await fetch(fundusImage).then(r => r.blob());
        formData.append('fundus', fundusBlob, 'fundus.jpg');
      }

      const timestamp = new Date().getTime();
      const random = Math.random().toString(36).substring(7);
      const response = await fetch(`http://localhost:5002/predict/${timestamp}/${random}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      setPrediction({
        severity: data.severity,
        confidence: data.confidence,
        severity_level: data.severity_level,
        probabilities: data.probabilities
      });
    } catch (error) {
      console.error('Error:', error);
      setError(error.message || 'An error occurred while making the prediction');
    } finally {
      setLoading(false);
    }
  };

  const renderPrediction = () => {
    if (!prediction) return null;
    
    // Get severity class for styling
    const severityClass = prediction.severity.toLowerCase().replace(' ', '-');
    
    return (
      <div className="prediction-container">
        <div className="main-prediction">
          <div className="prediction-label">Prediction</div>
          <div className={`prediction-value ${severityClass}`}>
            {prediction.severity}
          </div>
        </div>
      </div>
    );
  };

  const renderUploadSection = (title, image, setImage) => (
    <div className={`upload-section ${image ? 'active' : ''}`}>
      <h2>
        <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        {title}
      </h2>
      <div 
        className={`file-input-wrapper ${image ? 'has-image' : ''} ${dragOver ? 'drag-over' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={(e) => handleDrop(e, setImage)}
      >
        <input
          type="file"
          accept="image/*"
          onChange={(e) => handleImageUpload(e, setImage)}
          className="file-input"
        />
        {!image && (
          <div className="upload-placeholder">
            <svg className="upload-icon-large" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <span>Drag & drop or click to upload</span>
            <span className="file-info">Supports: JPG, PNG</span>
          </div>
        )}
        {image && (
          <div className="preview-container">
            <img src={image} alt="Preview" className="preview-image" />
            <button 
              className="remove-image"
              onClick={() => removeImage(setImage)}
              title="Remove image"
            >
              ×
            </button>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="App">
      <header className="App-header">
        <h1>Diabetic Retinopathy Detection</h1>
        
        <div className="upload-container">
          {renderUploadSection('OCT Image', octImage, setOctImage)}
          {renderUploadSection('Fundus Image', fundusImage, setFundusImage)}
        </div>

        {error && <div className="error-message">{error}</div>}

        <button 
          onClick={handlePredict} 
          disabled={loading || (!octImage && !fundusImage)}
          className="predict-button"
        >
          {loading ? 'Analyzing Images...' : 'Analyze Images'}
        </button>

        {loading && <div className="loading-spinner" />}

        {renderPrediction()}
      </header>
    </div>
  );
}

export default App; 