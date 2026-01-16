import { useState } from 'react';
import { Upload, Video, Wifi, CheckCircle2, Activity } from 'lucide-react';
import { ImageWithFallback } from '@/app/components/figma/ImageWithFallback';

type DetectionMode = 'image' | 'video' | 'realtime';
type Detection = {
  id: number;
  label: string;
  confidence: number;
  compliance: string;
  bbox: string;
};

export function DemoPage() {
  const [activeMode, setActiveMode] = useState<DetectionMode>('image');
  const [selectedModel, setSelectedModel] = useState('yolov8-v3');
  const [confidenceThreshold, setConfidenceThreshold] = useState(50);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);

  const models = [
    { value: 'yolov8-v3', label: 'YOLOv8 v3.2 (Recommended)' },
    { value: 'faster-rcnn', label: 'Faster R-CNN' },
    { value: 'efficientdet', label: 'EfficientDet-D4' }
  ];

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string);
        setDetections([]);
      };
      reader.readAsDataURL(file);
    }
  };

  const runDetection = () => {
    if (!uploadedImage) return;
    
    setIsProcessing(true);
    setDetections([]);
    
    // Simulate detection processing
    setTimeout(() => {
      const mockDetections: Detection[] = [
        {
          id: 1,
          label: 'Helmet',
          confidence: 96.8,
          compliance: 'COMPLIANT',
          bbox: '245, 120, 180, 160'
        },
        {
          id: 2,
          label: 'Motorcycle',
          confidence: 98.5,
          compliance: 'N/A',
          bbox: '150, 200, 400, 350'
        },
        {
          id: 3,
          label: 'Person',
          confidence: 97.2,
          compliance: 'N/A',
          bbox: '220, 100, 200, 380'
        }
      ];
      
      setDetections(mockDetections);
      setIsProcessing(false);
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-slate-50 pt-16">
      {/* Header */}
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <h1 className="text-4xl font-bold text-white mb-3">HelmetNet Detection System</h1>
          <p className="text-slate-300 text-lg">AI-powered helmet compliance detection</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex gap-6">
          {/* Left Sidebar - Configuration */}
          <div className="w-80 flex-shrink-0">
            <div className="bg-white rounded-xl border border-slate-200 shadow-lg sticky top-24">
              {/* Configuration Header */}
              <div className="p-6 border-b border-slate-200">
                <h2 className="font-semibold text-slate-900 text-lg">Configuration</h2>
              </div>

              {/* Model Settings */}
              <div className="p-6 border-b border-slate-200">
                <h3 className="text-sm font-semibold text-slate-700 mb-4">Model Settings</h3>
                
                <div className="mb-6">
                  <label className="text-sm text-slate-600 mb-2 block">Model Path</label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full bg-slate-50 text-slate-900 border border-slate-300 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:border-amber-500 focus:ring-2 focus:ring-amber-200 transition-all"
                  >
                    {models.map((model) => (
                      <option key={model.value} value={model.value}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-sm text-slate-600 mb-2 block">
                    Confidence Threshold
                    <span className="float-right text-amber-600 font-bold">
                      {confidenceThreshold}%
                    </span>
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
                  />
                  <div className="flex justify-between text-xs text-slate-500 mt-1">
                    <span>0%</span>
                    <span>100%</span>
                  </div>
                </div>
              </div>

              {/* Session Stats */}
              <div className="p-6">
                <h3 className="text-sm font-semibold text-slate-700 mb-4">Session Stats</h3>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Total Detections</span>
                    <span className="text-slate-900 font-semibold">{detections.length}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Model Status</span>
                    <span className="text-green-600 font-semibold flex items-center gap-1">
                      <span className="size-2 bg-green-500 rounded-full"></span>
                      Loaded
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Content Area */}
          <div className="flex-1">
            {/* Mode Tabs */}
            <div className="bg-white rounded-xl border border-slate-200 p-1.5 flex gap-2 mb-6 shadow-md">
              <button
                onClick={() => setActiveMode('image')}
                className={`flex-1 py-3 px-4 rounded-lg flex items-center justify-center gap-2 transition-all font-medium ${
                  activeMode === 'image'
                    ? 'bg-amber-500 text-slate-900 shadow-lg'
                    : 'text-slate-600 hover:bg-slate-50'
                }`}
              >
                <Upload className="size-5" />
                Image Detection
              </button>
              <button
                onClick={() => setActiveMode('video')}
                className={`flex-1 py-3 px-4 rounded-lg flex items-center justify-center gap-2 transition-all font-medium ${
                  activeMode === 'video'
                    ? 'bg-amber-500 text-slate-900 shadow-lg'
                    : 'text-slate-600 hover:bg-slate-50'
                }`}
              >
                <Video className="size-5" />
                Video Detection
              </button>
              <button
                onClick={() => setActiveMode('realtime')}
                className={`flex-1 py-3 px-4 rounded-lg flex items-center justify-center gap-2 transition-all font-medium ${
                  activeMode === 'realtime'
                    ? 'bg-amber-500 text-slate-900 shadow-lg'
                    : 'text-slate-600 hover:bg-slate-50'
                }`}
              >
                <Wifi className="size-5" />
                Real Time Detection
              </button>
            </div>

            {/* Image Detection Mode */}
            {activeMode === 'image' && (
              <div className="space-y-6">
                {/* Upload Section */}
                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-lg">
                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <h3 className="text-xl font-semibold text-slate-900 mb-2">Upload an Image</h3>
                      <p className="text-sm text-slate-600">
                        Supported formats: JPG, PNG, BMP
                      </p>
                    </div>
                    <div className="text-right bg-slate-50 px-4 py-3 rounded-lg">
                      <div className="text-xs text-slate-500 mb-1 font-medium">Quick Tips</div>
                      <div className="text-xs text-slate-700">Clear, well-lit images</div>
                      <div className="text-xs text-slate-700">Max size: 10MB</div>
                    </div>
                  </div>

                  <div className="border-2 border-dashed border-slate-300 rounded-xl p-12 text-center hover:border-amber-500 hover:bg-amber-50/30 transition-all">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                      id="file-upload"
                    />
                    <label
                      htmlFor="file-upload"
                      className="cursor-pointer flex flex-col items-center"
                    >
                      <Upload className="size-14 text-slate-400 mb-4" />
                      <span className="text-slate-700 mb-2 font-medium text-lg">
                        Drag and drop file here
                      </span>
                      <span className="text-sm text-slate-500 mb-4">
                        Image: JPEG/PNG, Max size: 10MB
                      </span>
                      <button className="bg-amber-500 text-slate-900 px-8 py-3 rounded-xl font-semibold hover:bg-amber-400 transition-all shadow-md hover:shadow-lg">
                        Browse Files
                      </button>
                    </label>
                  </div>

                  {uploadedImage && (
                    <div className="mt-6">
                      <button
                        onClick={runDetection}
                        disabled={isProcessing}
                        className={`w-full py-3 rounded-xl font-semibold transition-all shadow-md ${
                          isProcessing
                            ? 'bg-slate-300 text-slate-500 cursor-not-allowed'
                            : 'bg-amber-500 text-slate-900 hover:bg-amber-400 hover:shadow-lg'
                        }`}
                      >
                        {isProcessing ? 'Processing...' : 'Run Detection'}
                      </button>
                    </div>
                  )}
                </div>

                {/* Image Display */}
                {uploadedImage && (
                  <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-lg">
                    <h3 className="text-lg font-semibold text-slate-900 mb-4">Uploaded Image</h3>
                    <div className="bg-slate-50 rounded-xl overflow-hidden border border-slate-200 relative">
                      <img
                        src={uploadedImage}
                        alt="Uploaded"
                        className="w-full h-auto max-h-96 object-contain"
                      />
                      {isProcessing && (
                        <div className="absolute inset-0 bg-slate-900/60 flex items-center justify-center">
                          <div className="text-center">
                            <Activity className="size-16 text-amber-400 animate-spin mx-auto mb-4" />
                            <div className="text-white text-xl font-semibold">Analyzing Image...</div>
                            <div className="text-slate-300 mt-2">Processing with AI algorithms</div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Detection Results */}
                <div className="bg-white rounded-xl border border-slate-200 shadow-lg">
                  <div className="p-6 border-b border-slate-200 flex items-center justify-between">
                    <h3 className="text-xl font-semibold text-slate-900">Results</h3>
                    <span className="text-sm text-slate-600 bg-slate-100 px-3 py-1 rounded-lg">Model: HelmetNet</span>
                  </div>

                  <div className="p-6">
                    <div className="mb-6">
                      <h4 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
                        Detections Table
                        <span className="text-xs text-slate-500 font-normal">Sorted by confidence</span>
                      </h4>
                      
                      {detections.length > 0 ? (
                        <div className="overflow-x-auto border border-slate-200 rounded-xl">
                          <table className="w-full text-sm">
                            <thead className="bg-slate-50">
                              <tr className="border-b border-slate-200">
                                <th className="text-left py-3 px-4 font-semibold text-slate-700">#</th>
                                <th className="text-left py-3 px-4 font-semibold text-slate-700">LABEL</th>
                                <th className="text-left py-3 px-4 font-semibold text-slate-700">CONFIDENCE</th>
                                <th className="text-left py-3 px-4 font-semibold text-slate-700">COMPLIANCE</th>
                                <th className="text-left py-3 px-4 font-semibold text-slate-700">BBOX (X,Y,W,H)</th>
                              </tr>
                            </thead>
                            <tbody>
                              {detections.map((detection) => (
                                <tr key={detection.id} className="border-b border-slate-100 hover:bg-slate-50">
                                  <td className="py-3 px-4 text-slate-600">{detection.id}</td>
                                  <td className="py-3 px-4 text-slate-900 font-medium">{detection.label}</td>
                                  <td className="py-3 px-4">
                                    <span className="text-green-600 font-semibold">{detection.confidence}%</span>
                                  </td>
                                  <td className="py-3 px-4">
                                    {detection.compliance === 'COMPLIANT' ? (
                                      <span className="inline-flex items-center gap-1 text-green-600 font-medium">
                                        <CheckCircle2 className="size-4" />
                                        {detection.compliance}
                                      </span>
                                    ) : (
                                      <span className="text-slate-500">{detection.compliance}</span>
                                    )}
                                  </td>
                                  <td className="py-3 px-4 text-slate-600 font-mono text-xs">
                                    {detection.bbox}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <div className="text-center py-12 border border-slate-200 rounded-xl bg-slate-50">
                          <p className="text-slate-500">No results yet. Upload an image and click "Run detection".</p>
                        </div>
                      )}
                    </div>

                    <div className="bg-amber-50 border border-amber-200 p-5 rounded-xl text-sm text-slate-700">
                      <p className="mb-2">
                        <strong className="text-slate-900">Integration approach:</strong> Replace mock generation with a backend endpoint
                        returning <code className="bg-white border border-amber-300 px-2 py-0.5 rounded text-slate-800">{'{ detections: [{ label, conf, x, y, w, h }] }'}</code>.
                      </p>
                      <p className="text-slate-600">
                        Tip: For your final demo, add "export report" (model version, threshold, timestamp, detections).
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Video Detection Mode */}
            {activeMode === 'video' && (
              <div className="bg-white rounded-xl border border-slate-200 p-16 shadow-lg">
                <div className="text-center">
                  <Video className="size-20 text-slate-300 mx-auto mb-4" />
                  <h3 className="text-2xl font-semibold text-slate-900 mb-3">Video Detection</h3>
                  <p className="text-slate-600 mb-6 max-w-md mx-auto">
                    Upload a video file to process frame-by-frame helmet detection
                  </p>
                  <button className="bg-amber-500 text-slate-900 px-8 py-3 rounded-xl font-semibold hover:bg-amber-400 transition-all shadow-md">
                    Coming Soon
                  </button>
                </div>
              </div>
            )}

            {/* Real-time Detection Mode */}
            {activeMode === 'realtime' && (
              <div className="bg-white rounded-xl border border-slate-200 p-16 shadow-lg">
                <div className="text-center">
                  <Wifi className="size-20 text-slate-300 mx-auto mb-4" />
                  <h3 className="text-2xl font-semibold text-slate-900 mb-3">Real Time Detection</h3>
                  <p className="text-slate-600 mb-6 max-w-md mx-auto">
                    Connect to a webcam or RTSP stream for live helmet detection
                  </p>
                  <button className="bg-amber-500 text-slate-900 px-8 py-3 rounded-xl font-semibold hover:bg-amber-400 transition-all shadow-md">
                    Coming Soon
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
