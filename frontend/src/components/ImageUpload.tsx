import React, { useState } from 'react';
import { Box, Button, Typography, CircularProgress } from '@mui/material';
import { styled } from '@mui/system';

const UploadBox = styled(Box)({
  border: '2px dashed #ccc',
  borderRadius: '8px',
  padding: '20px',
  textAlign: 'center',
  marginBottom: '20px',
  cursor: 'pointer',
  '&:hover': {
    borderColor: '#1976d2',
  },
});

const PreviewImage = styled('img')({
  maxWidth: '100%',
  maxHeight: '300px',
  marginTop: '10px',
  borderRadius: '4px',
});

interface ImageUploadProps {
  onImagesSelected: (octImage: File | null, fundusImage: File | null) => void;
  loading: boolean;
  disabled: boolean;
}

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_SIZE = 5 * 1024 * 1024; // 5MB

const ImageUpload: React.FC<ImageUploadProps> = ({ onImagesSelected, loading, disabled }) => {
  const [octImage, setOctImage] = useState<File | null>(null);
  const [fundusImage, setFundusImage] = useState<File | null>(null);
  const [octPreview, setOctPreview] = useState<string>('');
  const [fundusPreview, setFundusPreview] = useState<string>('');
  const [error, setError] = useState<string>('');

  const validateFile = (file: File): string | null => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return 'Please upload a JPEG or PNG image.';
    }
    if (file.size > MAX_SIZE) {
      return 'File size must be less than 5MB.';
    }
    return null;
  };

  const handleImageUpload = (file: File, type: 'oct' | 'fundus') => {
    console.log(`Handling ${type} image upload:`, file.name, file.type, file.size);
    
    const error = validateFile(file);
    if (error) {
      setError(error);
      return;
    }

    // Create a preview URL
    const previewUrl = URL.createObjectURL(file);
    
    if (type === 'oct') {
      setOctImage(file);
      setOctPreview(previewUrl);
    } else {
      setFundusImage(file);
      setFundusPreview(previewUrl);
    }
    
    setError('');
  };

  const handleOctUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleImageUpload(file, 'oct');
    }
  };

  const handleFundusUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleImageUpload(file, 'fundus');
    }
  };

  const handleSubmit = () => {
    if (octImage || fundusImage) {
      console.log('Submitting images for analysis:');
      if (octImage) console.log('OCT:', octImage.name, octImage.type, octImage.size);
      if (fundusImage) console.log('Fundus:', fundusImage.name, fundusImage.type, fundusImage.size);
      onImagesSelected(octImage, fundusImage);
    } else {
      setError('Please upload at least one image (OCT or Fundus)');
    }
  };

  return (
    <Box>
      <Box display="flex" gap={2}>
        <Box flex={1}>
          <Typography variant="h6" gutterBottom>
            OCT Image (Optional)
          </Typography>
          <input
            accept="image/jpeg,image/png,image/webp"
            type="file"
            id="oct-upload"
            hidden
            onChange={handleOctUpload}
            disabled={loading || disabled}
          />
          <label htmlFor="oct-upload">
            <UploadBox sx={{ opacity: disabled ? 0.5 : 1 }}>
              {octPreview ? (
                <PreviewImage src={octPreview} alt="OCT preview" />
              ) : (
                <Typography>
                  Click or drag to upload OCT image (JPEG/PNG, max 5MB)
                </Typography>
              )}
            </UploadBox>
          </label>
        </Box>

        <Box flex={1}>
          <Typography variant="h6" gutterBottom>
            Fundus Image (Optional)
          </Typography>
          <input
            accept="image/jpeg,image/png,image/webp"
            type="file"
            id="fundus-upload"
            hidden
            onChange={handleFundusUpload}
            disabled={loading || disabled}
          />
          <label htmlFor="fundus-upload">
            <UploadBox sx={{ opacity: disabled ? 0.5 : 1 }}>
              {fundusPreview ? (
                <PreviewImage src={fundusPreview} alt="Fundus preview" />
              ) : (
                <Typography>
                  Click or drag to upload fundus image (JPEG/PNG, max 5MB)
                </Typography>
              )}
            </UploadBox>
          </label>
        </Box>
      </Box>

      {error && (
        <Typography color="error" textAlign="center" mt={2}>
          {error}
        </Typography>
      )}

      <Box display="flex" justifyContent="center" mt={3}>
        <Button
          variant="contained"
          onClick={handleSubmit}
          disabled={(!octImage && !fundusImage) || loading || disabled}
        >
          {loading ? (
            <CircularProgress size={24} color="inherit" />
          ) : (
            'Analyze Images'
          )}
        </Button>
      </Box>
    </Box>
  );
};

export default ImageUpload; 