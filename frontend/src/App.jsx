import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const API_URL = 'http://localhost:8000'

const initialFeatures = {
  radius_mean: '', texture_mean: '', perimeter_mean: '', area_mean: '', smoothness_mean: '', compactness_mean: '', concavity_mean: '', concave_points_mean: '', symmetry_mean: '', fractal_dimension_mean: '',
  radius_se: '', texture_se: '', perimeter_se: '', area_se: '', smoothness_se: '', compactness_se: '', concavity_se: '', concave_points_se: '', symmetry_se: '', fractal_dimension_se: '',
  radius_worst: '', texture_worst: '', perimeter_worst: '', area_worst: '', smoothness_worst: '', compactness_worst: '', concavity_worst: '', concave_points_worst: '', symmetry_worst: '', fractal_dimension_worst: ''
}

const mockMalignant = {
  radius_mean: 17.99, texture_mean: 10.38, perimeter_mean: 122.8, area_mean: 1001.0, smoothness_mean: 0.1184, compactness_mean: 0.2776, concavity_mean: 0.3001, concave_points_mean: 0.1471, symmetry_mean: 0.2419, fractal_dimension_mean: 0.07871,
  radius_se: 1.095, texture_se: 0.9053, perimeter_se: 8.589, area_se: 153.4, smoothness_se: 0.006399, compactness_se: 0.04904, concavity_se: 0.05373, concave_points_se: 0.01587, symmetry_se: 0.03003, fractal_dimension_se: 0.006193,
  radius_worst: 25.38, texture_worst: 17.33, perimeter_worst: 184.6, area_worst: 2019.0, smoothness_worst: 0.1622, compactness_worst: 0.6656, concavity_worst: 0.7119, concave_points_worst: 0.2654, symmetry_worst: 0.4601, fractal_dimension_worst: 0.1189
}

const mockBenign = {
  radius_mean: 13.54, texture_mean: 14.36, perimeter_mean: 87.46, area_mean: 566.3, smoothness_mean: 0.09779, compactness_mean: 0.08129, concavity_mean: 0.06664, concave_points_mean: 0.04781, symmetry_mean: 0.1885, fractal_dimension_mean: 0.05766,
  radius_se: 0.2699, texture_se: 0.7886, perimeter_se: 2.058, area_se: 23.56, smoothness_se: 0.008462, compactness_se: 0.0146, concavity_se: 0.02387, concave_points_se: 0.01315, symmetry_se: 0.0198, fractal_dimension_se: 0.0023,
  radius_worst: 15.11, texture_worst: 19.26, perimeter_worst: 99.7, area_worst: 711.2, smoothness_worst: 0.144, compactness_worst: 0.1773, concavity_worst: 0.239, concave_points_worst: 0.1288, symmetry_worst: 0.2977, fractal_dimension_worst: 0.07259
}

function App() {
  const [activeTab, setActiveTab] = useState('live')
  
  // Live Prediction State
  const [features, setFeatures] = useState(initialFeatures)
  const [liveLoading, setLiveLoading] = useState(false)
  const [liveResult, setLiveResult] = useState(null)
  const [liveError, setLiveError] = useState(null)
  const [explainResult, setExplainResult] = useState(null)

  // History State
  const [history, setHistory] = useState([])
  const [historyLoading, setHistoryLoading] = useState(false)

  const fetchHistory = async () => {
    setHistoryLoading(true)
    try {
      const response = await fetch(`${API_URL}/history`)
      if (response.ok) {
        const data = await response.json()
        setHistory(data)
      }
    } catch (err) {
      console.error("Failed to fetch history", err)
    } finally {
      setHistoryLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'history') {
      fetchHistory()
    }
  }, [activeTab])

  // Batch Prediction State
  const [batchFile, setBatchFile] = useState(null)
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchResults, setBatchResults] = useState(null)
  const [batchError, setBatchError] = useState(null)
  const fileInputRef = useRef(null)

  const handleFeatureChange = (e) => {
    const { name, value } = e.target
    setFeatures(prev => ({ ...prev, [name]: value }))
  }

  const handleRandomSample = (type) => {
    if (type === 'malignant') setFeatures(mockMalignant)
    else setFeatures(mockBenign)
  }

  const handleLiveSubmit = async (e) => {
    e.preventDefault()
    setLiveLoading(true)
    setLiveError(null)
    setLiveResult(null)
    setExplainResult(null)

    try {
      const payload = {}
      for (const key in features) {
        payload[key] = parseFloat(features[key])
        if (isNaN(payload[key])) throw new Error(`Invalid value for ${key.replace('_', ' ')}`)
      }

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || 'Prediction failed')
      }

      const data = await response.json()
      setLiveResult(data)
      
      // Also fetch explainability
      fetch(`${API_URL}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }).then(res => res.json()).then(expData => {
        if(expData.feature_importance) {
          setExplainResult(expData.feature_importance)
        }
      }).catch(err => console.error("Explain error", err))

    } catch (err) {
      setLiveError(err.message)
    } finally {
      setLiveLoading(false)
    }
  }

  const handleBatchSubmit = async (e) => {
    e.preventDefault()
    if (!batchFile) return
    setBatchLoading(true)
    setBatchError(null)
    setBatchResults(null)

    const formData = new FormData()
    formData.append("file", batchFile)

    try {
      const response = await fetch(`${API_URL}/batch-predict`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || 'Batch prediction failed')
      }

      const data = await response.json()
      setBatchResults(data.batch_results)
    } catch (err) {
      setBatchError(err.message)
    } finally {
      setBatchLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>OncoVision AI</h1>
        <p>Advanced SVM-based Breast Cancer Classification</p>
      </header>

      <div className="tabs-container">
        <div className="tab-list">
          <button className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`} onClick={() => setActiveTab('live')}>Live Prediction</button>
          <button className={`tab-btn ${activeTab === 'batch' ? 'active' : ''}`} onClick={() => setActiveTab('batch')}>Batch Prediction</button>
          <button className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`} onClick={() => setActiveTab('analytics')}>Analytics</button>
          <button className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>History</button>
        </div>

        {activeTab === 'live' && (
          <motion.main 
            className="main-content"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <section className="glass-panel form-panel">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <h2 style={{ fontSize: '1.5rem', fontWeight: 600 }}>Tumor Features</h2>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button type="button" className="btn btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '0.875rem' }} onClick={() => handleRandomSample('benign')}>Load Benign</button>
                  <button type="button" className="btn btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '0.875rem' }} onClick={() => handleRandomSample('malignant')}>Load Malignant</button>
                </div>
              </div>
              
              <form onSubmit={handleLiveSubmit}>
                <div className="form-grid">
                  {Object.keys(features).map(key => (
                    <div className="input-group" key={key}>
                      <label htmlFor={key}>{key.replace(/_/g, ' ')}</label>
                      <input
                        type="number"
                        step="any"
                        id={key}
                        name={key}
                        value={features[key]}
                        onChange={handleFeatureChange}
                        required
                      />
                    </div>
                  ))}
                </div>

                <div className="actions">
                  <button 
                    type="button" 
                    className="btn btn-secondary" 
                    onClick={() => setFeatures(initialFeatures)}
                    disabled={liveLoading}
                  >
                    Clear
                  </button>
                  <button type="submit" className="btn btn-primary" disabled={liveLoading}>
                    {liveLoading ? <span className="spinner"></span> : 'Run Prediction'}
                  </button>
                </div>
              </form>
              {liveError && <div style={{ color: 'var(--danger-color)', marginTop: '1rem', padding: '1rem', background: 'rgba(239,68,68,0.1)', borderRadius: '8px' }}>{liveError}</div>}
            </section>

            <section className="glass-panel result-panel">
              {!liveResult ? (
                <div className="result-placeholder">
                  <div className="icon-placeholder">🔬</div>
                  <h3>Awaiting Data</h3>
                  <p style={{ textAlign: 'center' }}>Enter tumor features or load a sample to see the AI prediction.</p>
                </div>
              ) : (
                <div className="prediction-result" style={{width: '100%'}}>
                  <h2 style={{ fontSize: '1.25rem', color: 'var(--text-muted)', marginBottom: '1rem', textAlign: 'center' }}>AI Diagnosis</h2>
                  <div style={{ textAlign: 'center' }}>
                    <div className={`status-badge ${liveResult.prediction === 'Malignant' ? 'status-malignant' : 'status-benign'}`}>
                      {liveResult.prediction}
                    </div>
                  </div>
                  
                  <div style={{ marginTop: '2rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Confidence</span>
                      <span className="confidence-text">{liveResult.confidence}</span>
                    </div>
                    
                    <div className="confidence-bar-container">
                      <div 
                        className={`confidence-bar ${liveResult.prediction === 'Malignant' ? 'bar-malignant' : 'bar-benign'}`}
                        style={{ width: liveResult.confidence }}
                      ></div>
                    </div>
                  </div>

                  <div className="prob-details">
                    <div className="prob-item">
                      <span className="prob-label">Benign Prob.</span>
                      <span className="prob-value" style={{ color: 'var(--success-color)' }}>{liveResult.probabilities.Benign}</span>
                    </div>
                    <div className="prob-item" style={{ textAlign: 'right' }}>
                      <span className="prob-label">Malignant Prob.</span>
                      <span className="prob-value" style={{ color: 'var(--danger-color)' }}>{liveResult.probabilities.Malignant}</span>
                    </div>
                  </div>
                  
                  {explainResult && (
                    <div style={{marginTop: '2rem', textAlign: 'left', width: '100%'}}>
                      <h3 style={{fontSize: '1.1rem', marginBottom: '1rem', color: 'var(--primary-color)'}}>Top Contributing Features</h3>
                      {Object.entries(explainResult).slice(0, 5).map(([feature, val]) => (
                        <div key={feature} style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.9rem'}}>
                          <span style={{textTransform: 'capitalize'}}>{feature.replace(/_/g, ' ')}</span>
                          <span style={{color: val > 0 ? 'var(--danger-color)' : 'var(--success-color)'}}>
                            {val > 0 ? '↑' : '↓'} {Math.abs(val).toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </section>
          </motion.main>
        )}

        {activeTab === 'batch' && (
          <motion.div 
            className="glass-panel" 
            style={{maxWidth: '800px', margin: '0 auto'}}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <h2 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '1rem' }}>Batch Prediction</h2>
            <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>Upload a CSV file containing multiple tumor records. The file must contain the same 30 features as the training dataset.</p>
            
            <form onSubmit={handleBatchSubmit} style={{display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '2rem'}}>
              <input 
                type="file" 
                accept=".csv" 
                onChange={(e) => setBatchFile(e.target.files[0])}
                ref={fileInputRef}
                style={{padding: '0.5rem', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--glass-border)', borderRadius: '8px', flexGrow: 1}}
              />
              <button type="submit" className="btn btn-primary" disabled={batchLoading || !batchFile}>
                {batchLoading ? <span className="spinner"></span> : 'Predict Batch'}
              </button>
            </form>
            
            {batchError && <div style={{ color: 'var(--danger-color)', marginBottom: '1rem', padding: '1rem', background: 'rgba(239,68,68,0.1)', borderRadius: '8px' }}>{batchError}</div>}
            
            {batchResults && (
              <div style={{overflowX: 'auto'}}>
                <h3 style={{marginBottom: '1rem'}}>Results ({batchResults.length} records)</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Row Index</th>
                      <th>Prediction</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchResults.map((row) => (
                      <tr key={row.index}>
                        <td>{row.index}</td>
                        <td>
                           <span className={row.prediction === 'Malignant' ? 'status-malignant' : 'status-benign'} style={{padding: '0.2rem 0.6rem', borderRadius: '4px', fontSize: '0.8rem', fontWeight: 'bold'}}>
                             {row.prediction}
                           </span>
                        </td>
                        <td>{row.confidence}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {activeTab === 'analytics' && (
          <motion.div 
            className="metrics-container"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="metric-card">
              <h2 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '1rem' }}>Model Confusion Matrix</h2>
              <p style={{ color: 'var(--text-muted)' }}>Displays the true positives, false positives, true negatives, and false negatives from the best SVM model.</p>
              <img src={`${API_URL}/metrics/confusion-matrix`} alt="Confusion Matrix" onError={(e) => { e.target.onerror = null; e.target.src = ''; e.target.alt = 'Image not available. Did you train the model?'; }} />
            </div>
            
            <div className="metric-card">
              <h2 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '1rem' }}>PCA Decision Boundary</h2>
              <p style={{ color: 'var(--text-muted)' }}>Visualization of the SVM decision boundary reduced to 2 principal components.</p>
              <img src={`${API_URL}/metrics/pca-plot`} alt="PCA Plot" onError={(e) => { e.target.onerror = null; e.target.src = ''; e.target.alt = 'Image not available. Did you train the model?'; }} />
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <motion.div 
            className="glass-panel" 
            style={{maxWidth: '800px', margin: '0 auto'}}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h2 style={{ fontSize: '1.5rem', fontWeight: 600 }}>Prediction History</h2>
              <button onClick={fetchHistory} className="btn btn-secondary" style={{padding: '0.5rem 1rem', fontSize: '0.85rem'}}>Refresh</button>
            </div>
            
            {historyLoading ? (
              <div style={{textAlign: 'center', padding: '2rem'}}><span className="spinner"></span></div>
            ) : history.length === 0 ? (
              <p style={{ textAlign: 'center', color: 'var(--text-muted)' }}>No predictions recorded yet. Run a live prediction first!</p>
            ) : (
              <div style={{overflowX: 'auto'}}>
                <table>
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Prediction</th>
                      <th>Confidence</th>
                      <th>Top Feature</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((record) => (
                      <tr key={record.id}>
                        <td style={{fontSize: '0.85rem', color: 'var(--text-muted)'}}>{new Date(record.timestamp + 'Z').toLocaleString()}</td>
                        <td>
                           <span className={record.prediction === 'Malignant' ? 'status-malignant' : 'status-benign'} style={{padding: '0.2rem 0.6rem', borderRadius: '4px', fontSize: '0.8rem', fontWeight: 'bold'}}>
                             {record.prediction}
                           </span>
                        </td>
                        <td>{(record.confidence * 100).toFixed(2)}%</td>
                        <td style={{fontSize: '0.85rem'}}>
                          {record.top_feature_1_name ? (
                            <span style={{textTransform: 'capitalize'}}>{record.top_feature_1_name.replace(/_/g, ' ')} ({record.top_feature_1_val > 0 ? '+' : '-'})</span>
                          ) : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default App
