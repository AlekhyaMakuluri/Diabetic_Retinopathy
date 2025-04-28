import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  Typography,
  AppBar,
  Toolbar,
  Alert,
  Snackbar,
  CssBaseline,
  ThemeProvider,
  createTheme,
  CircularProgress,
  Button,
} from '@mui/material';
import ImageUpload from './components/ImageUpload';
import Results from './components/Results';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

interface Prediction {
  severity: string;
  confidence: number;
  severity_level: number;
  probabilities: number[];
}

const API_BASE_URL = 'http://localhost:5002';

function App() {
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'error'>('checking');
  const [debugInfo, setDebugInfo] = useState<string>('');

  useEffect(() => {
    // Check backend health on component mount
    const checkBackendHealth = async () => {
      try {
        setDebugInfo('Checking backend health...');
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        setDebugInfo(`Health check response: ${JSON.stringify(data)}`);
        
        if (response.ok && data.status === 'healthy') {
          setBackendStatus('connected');
        } else {
          setBackendStatus('error');
          setError('Backend server is not responding properly');
        }
      } catch (err) {
        setBackendStatus('error');
        setError('Cannot connect to backend server. Please make sure it is running.');
        setDebugInfo(`Connection error: ${err}`);
      }
    };

    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleImagesSelected = async (octImage: File | null, fundusImage: File | null) => {
    if (backendStatus !== 'connected') {
      setError('Backend server is not available. Please try again later.');
      return;
    }

    try {
      setLoading(true);
      setError('');
      setPrediction(null);
      setDebugInfo('Starting image upload process...');

      if (!octImage || !fundusImage) {
        throw new Error('Please upload both OCT and fundus images');
      }

      setDebugInfo(`Preparing form data with images:\nOCT: ${octImage.name} (${octImage.size} bytes)\nFundus: ${fundusImage.name} (${fundusImage.size} bytes)`);
      const formData = new FormData();
      formData.append('oct', octImage);
      formData.append('fundus', fundusImage);
      formData.append('timestamp', new Date().getTime().toString());
      formData.append('random', Math.random().toString(36).substring(7));

      const timestamp = new Date().getTime();
      const random = Math.random().toString(36).substring(7);
      const url = `${API_BASE_URL}/predict/${timestamp}/${random}`;

      setDebugInfo(`Sending request to: ${url}`);
      console.log('Sending request to:', url);
      console.log('Form data:', Object.fromEntries(formData.entries()));

      try {
        const response = await fetch(url, {
          method: 'POST',
          body: formData,
          headers: {
            'Accept': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
          },
          mode: 'cors',
          credentials: 'same-origin'
        });

        console.log('Response received:', response);
        console.log('Response status:', response.status);
        console.log('Response headers:', Object.fromEntries(response.headers.entries()));
        
        setDebugInfo(`Response status: ${response.status}\nResponse headers: ${JSON.stringify(Object.fromEntries(response.headers.entries()))}`);
        
        if (!response.ok) {
          const errorData = await response.json();
          console.error('Error response:', errorData);
          setDebugInfo(`Error response: ${JSON.stringify(errorData)}`);
          throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        const result = await response.json();
        console.log('Response data:', result);
        setDebugInfo(`Response data: ${JSON.stringify(result)}`);

        if (!result.severity || !result.confidence) {
          setDebugInfo('Invalid response format received');
          throw new Error('Invalid response format from server');
        }

        setPrediction(result);
        setDebugInfo('Prediction set successfully');
      } catch (err) {
        console.error('Error during prediction:', err);
        setError(err instanceof Error ? err.message : 'Failed to analyze images. Please try again.');
        setPrediction(null);
        setDebugInfo(`Error: ${err}`);
      }
    } catch (err) {
      console.error('Error during prediction:', err);
      setError(err instanceof Error ? err.message : 'Failed to analyze images. Please try again.');
      setPrediction(null);
      setDebugInfo(`Error: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Diabetic Retinopathy Detection
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="body2">
                Backend Status:
              </Typography>
              <Typography 
                variant="body2" 
                sx={{ 
                  color: backendStatus === 'connected' ? 'success.main' : 
                         backendStatus === 'checking' ? 'warning.main' : 'error.main'
                }}
              >
                {backendStatus === 'checking' ? 'Checking...' : 
                 backendStatus === 'connected' ? 'Connected' : 'Disconnected'}
              </Typography>
            </Box>
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ mt: 4 }}>
          <Box textAlign="center" mb={4}>
            <Typography variant="h4" gutterBottom>
              Early Detection of Diabetic Retinopathy
            </Typography>
            {/* <Typography variant="body1" color="text.secondary">
              Upload both OCT and fundus images for analysis
            </Typography> */}
          </Box>

          <ImageUpload 
            onImagesSelected={handleImagesSelected} 
            loading={loading} 
            disabled={backendStatus !== 'connected'}
          />
          
          {loading && (
            <Box display="flex" justifyContent="center" mt={4}>
              <CircularProgress />
            </Box>
          )}

          {/* Debug Information Box */}
          <Box sx={{ mt: 2, p: 2, backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
            <Typography variant="h6" gutterBottom>
              Debug Information (Version 1.0.1)
            </Typography>
            <Typography variant="body2" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {debugInfo}
            </Typography>
          </Box>
          
          {prediction && (
            <Box sx={{ mt: 2, p: 2, backgroundColor: '#e3f2fd', borderRadius: '8px' }}>
              <Typography variant="h6" gutterBottom>
                Prediction Results
              </Typography>
              <Typography variant="body1">
                Severity: {prediction.severity}
              </Typography>
              <Typography variant="body1">
                Confidence: {(prediction.confidence * 100).toFixed(2)}%
              </Typography>
              <Typography variant="body1">
                Level: {prediction.severity_level}
              </Typography>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={() => setPrediction(null)}
                sx={{ mt: 2 }}
              >
                Clear Results
              </Button>
            </Box>
          )}

          <Snackbar
            open={!!error}
            autoHideDuration={6000}
            onClose={() => setError('')}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
          >
            <Alert 
              severity="error" 
              onClose={() => setError('')}
              variant="filled"
            >
              {error}
            </Alert>
          </Snackbar>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App; 