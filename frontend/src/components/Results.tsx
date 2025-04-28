import React from 'react';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Chip,
} from '@mui/material';
import { styled } from '@mui/system';

const ResultPaper = styled(Paper)({
  padding: '24px',
  marginTop: '24px',
  borderRadius: '12px',
  backgroundColor: '#f5f5f5',
});

const getSeverityColor = (level: number): string => {
  const colors = ['success', 'info', 'warning', 'error', 'error'];
  return colors[level] || 'default';
};

interface ResultsProps {
  prediction: {
    severity: string;
    confidence: number;
    severity_level: number;
    probabilities: number[];
  };
}

const Results: React.FC<ResultsProps> = ({ prediction }) => {
  console.log('Results component rendering with prediction:', prediction);
  console.log('Prediction type:', typeof prediction);
  console.log('Prediction keys:', Object.keys(prediction));
  
  if (!prediction || typeof prediction !== 'object') {
    console.error('Invalid prediction data:', prediction);
    return (
      <ResultPaper elevation={3}>
        <Typography variant="h5" color="error">
          Error: Invalid prediction data
        </Typography>
      </ResultPaper>
    );
  }

  const severityLevels = ['Normal', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR'];
  const confidencePercent = (prediction.confidence * 100).toFixed(2);
  const severityColor = getSeverityColor(prediction.severity_level);

  console.log('Calculated values:', {
    severityLevels,
    confidencePercent,
    severityColor
  });

  return (
    <ResultPaper elevation={3}>
      <Box sx={{ mb: 2, p: 2, backgroundColor: '#e3f2fd', borderRadius: '8px' }}>
        <Typography variant="body2" color="text.secondary">
          Debug Info:
        </Typography>
        <Typography variant="body2">
          Severity: {prediction.severity}
        </Typography>
        <Typography variant="body2">
          Confidence: {confidencePercent}%
        </Typography>
        <Typography variant="body2">
          Level: {prediction.severity_level}
        </Typography>
      </Box>

      <Typography variant="h5" gutterBottom>
        Analysis Results
      </Typography>

      <Box mt={3}>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6">
            Severity Level:
          </Typography>
          <Chip
            label={prediction.severity}
            color={severityColor as any}
            size="medium"
          />
        </Box>

        <Box mt={4}>
          <Typography variant="body1" color="text.secondary">
            {prediction.severity_level === 0 ? (
              "No signs of diabetic retinopathy detected. Regular check-ups are recommended."
            ) : (
              "Please consult with an eye care professional for a thorough examination and treatment plan."
            )}
          </Typography>
        </Box>
      </Box>

      <Box sx={{ mt: 4 }}>
        <Typography variant="h6" color="primary">
          Predicted Severity: {prediction.severity}
        </Typography>
        <Typography variant="body1">
          Confidence: {confidencePercent}%
        </Typography>
      </Box>

      <Box sx={{ mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Probability Distribution
        </Typography>
        {prediction.probabilities.map((prob, index) => (
          <Box key={index} sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="body2">
                {severityLevels[index]}
              </Typography>
              <Typography variant="body2">
                {(prob * 100).toFixed(2)}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={prob * 100}
              sx={{
                height: 10,
                borderRadius: 5,
                backgroundColor: '#e0e0e0',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: index === prediction.severity_level ? '#4caf50' : '#2196f3',
                },
              }}
            />
          </Box>
        ))}
      </Box>
    </ResultPaper>
  );
};

export default Results; 