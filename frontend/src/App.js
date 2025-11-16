import React, { useState, useEffect } from 'react';
import { Upload, Brain, Activity, AlertCircle, CheckCircle, Layers, Users } from 'lucide-react';
import TrainingPage from './components/TrainingPage';

const API_URL = 'http://localhost:5000/api';

export default function BrainTumorSegmentation() {
  const [currentPage, setCurrentPage] = useState('segmentation');
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [file, setFile] = useState(null);
  const [maskFile, setMaskFile] = useState(null);
  const [sliceIdx, setSliceIdx] = useState(0);
  const [maxSlices, setMaxSlices] = useState(0);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);
  useEffect(() => {
    if (file && maskFile && maxSlices > 0) {
      const mid = Math.floor(maxSlices / 2);
      setSliceIdx(mid);
    }
  }, [file, maskFile, maxSlices]);

  const fetchModels = async () => {
    try {
      const response = await fetch(`${API_URL}/models`);
      const data = await response.json();
      setModels(data.models);
      if (data.models.length > 0) {
        setSelectedModel(data.models[0].name);
      }
    } catch (err) {
      console.error('Error fetching models:', err);
      setError('Failed to fetch models. Is the API server running?');
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.name.endsWith('.npy')) {
      setFile(selectedFile);
      setError(null);
    } else setError('Please select a .npy file');
  };

  const handleMaskChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.name.endsWith('.npy')) {
      setMaskFile(selectedFile);
      setError(null);
    } else setError('Please select a .npy file for mask');
  };

  const handleSegment = async () => {
    if (!file || !maskFile || !selectedModel) {
      setError('Please select both image file, mask file, and model');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('image', file);
    formData.append('mask', maskFile);
    formData.append('model', selectedModel);
    formData.append('slice_idx', sliceIdx.toString());

    try {
      const response = await fetch(`${API_URL}/segment`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResults(data);
        setMaxSlices(data.info.total_slices - 1);
        if (sliceIdx >= data.info.total_slices) {
          setSliceIdx(Math.floor(data.info.total_slices / 2));
        }
      } else {
        setError(data.error || 'Segmentation failed');
      }
    } catch (err) {
      setError('Failed to connect to server. Make sure api_server.py is running on port 5000.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSliceChange = (newSlice) => {
    setSliceIdx(newSlice);
    if (results && file && maskFile) {
      handleSegment();
    }
  };

  return (
    <>
      {/* NAVIGATION BAR */}
      <nav className="bg-white shadow-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <Brain className="w-8 h-8 text-indigo-600" />
              <span className="text-xl font-bold text-gray-800">Federated Brain Segmentation</span>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => setCurrentPage('training')}
                className={`px-4 py-2 rounded-lg transition-all flex items-center gap-2 ${
                  currentPage === 'training'
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Users className="w-4 h-4" />
                Training
              </button>

              <button
                onClick={() => setCurrentPage('segmentation')}
                className={`px-4 py-2 rounded-lg transition-all flex items-center gap-2 ${
                  currentPage === 'segmentation'
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Brain className="w-4 h-4" />
                Segmentation
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* TRAINING PAGE */}
      {currentPage === 'training' && <TrainingPage />}

      {/* SEGMENTATION PAGE */}
      {currentPage === 'segmentation' && (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
          {/* HEADER */}
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white py-8 shadow-lg">
            <div className="max-w-6xl mx-auto px-4">
              <div className="flex items-center gap-3 mb-2">
                <Brain className="w-10 h-10" />
                <h1 className="text-4xl font-bold">Brain Tumor Segmentation</h1>
              </div>
              <p className="text-blue-100 text-lg">
                Federated Learning Powered Medical AI - Privacy-Preserving Brain Tumor Detection
              </p>
            </div>
          </div>

          {/* ------- YOUR ENTIRE SEGMENTATION UI HERE ------- */}
          <div className="max-w-6xl mx-auto px-4 py-8">

            {/* Upload Image */}
            <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <Upload className="w-6 h-6" />
                Upload & Configure
              </h2>

              {/* MRI File */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Brain MRI Image File (.npy)
                </label>
                <div className="border-2 border-dashed rounded-lg p-6 text-center border-gray-300 hover:border-blue-400 transition-all">
                  <Upload className="w-10 h-10 mx-auto mb-2 text-gray-400" />
                  <p className="text-md font-medium text-gray-700 mb-2">
                    {file ? file.name : 'Select image .npy file'}
                  </p>
                  <label className="inline-block px-4 py-2 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700">
                    Browse Image File
                    <input type="file" accept=".npy" onChange={handleFileChange} className="hidden" />
                  </label>
                </div>
              </div>

              {/* MASK FILE */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Ground Truth Mask File (.npy)
                </label>
                <div className="border-2 border-dashed rounded-lg p-6 text-center border-gray-300 hover:border-green-400 transition-all">
                  <Upload className="w-10 h-10 mx-auto mb-2 text-gray-400" />
                  <p className="text-md font-medium text-gray-700 mb-2">
                    {maskFile ? maskFile.name : 'Select mask .npy file'}
                  </p>
                  <label className="inline-block px-4 py-2 bg-green-600 text-white rounded-lg cursor-pointer hover:bg-green-700">
                    Browse Mask File
                    <input type="file" accept=".npy" onChange={handleMaskChange} className="hidden" />
                  </label>
                </div>
              </div>

              {/* SELECT MODEL */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Global Model
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full px-4 py-2 border rounded-lg"
                >
                  {models.map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.display_name || model.name} ({model.size_mb} MB)
                    </option>
                  ))}
                </select>
              </div>

              {/* SLICE SLIDER */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                  <Layers className="w-4 h-4" />
                  Select Z-Slice (Layer): {sliceIdx} / {maxSlices}
                </label>
                <input
                  type="range"
                  min="0"
                  max={maxSlices}
                  value={sliceIdx}
                  onChange={(e) => handleSliceChange(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg"
                />
              </div>

              {/* BUTTON */}
              <button
                onClick={handleSegment}
                disabled={loading || !file || !maskFile}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-lg"
              >
                {loading ? 'Processing...' : 'Segment Image'}
              </button>

              {/* ERROR */}
              {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex gap-2">
                  <AlertCircle className="w-5 h-5 text-red-600" />
                  <p className="text-red-800 text-sm">{error}</p>
                </div>
              )}
            </div>

            {/* RESULTS */}
            {results && (
              <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                  Segmentation Results
                </h2>

                <div className="mb-6">
                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-lg border">
                    <h3 className="font-semibold flex items-center gap-2 mb-2">
                      <Activity className="w-5 h-5 text-blue-600" />
                      Dice Score
                    </h3>
                    <p className="text-4xl font-bold text-blue-600">
                      {results.metrics.dice_percentage}%
                    </p>
                  </div>
                </div>

                <div className="border rounded-lg p-4">
                  <h3 className="font-semibold flex items-center gap-2 mb-2">
                    <Layers className="w-5 h-5" />
                    Visual Comparison — Slice {results.info.slice_idx} / {results.info.total_slices - 1}
                  </h3>
                  <img
                    src={`data:image/png;base64,${results.visualization}`}
                    className="w-full rounded-lg"
                    alt="Segmentation"
                  />
                </div>
              </div>
            )}

          </div>
        </div>
      )}
    </>
  );
}
