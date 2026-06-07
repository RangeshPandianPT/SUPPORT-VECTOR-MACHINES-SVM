import { useState } from 'react'

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
  const [features, setFeatures] = useState(initialFeatures)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    const { name, value } = e.target
    setFeatures(prev => ({ ...prev, [name]: value }))
  }

  const handleRandomSample = (type) => {
    if (type === 'malignant') setFeatures(mockMalignant)
    else setFeatures(mockBenign)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Convert all to floats
      const payload = {}
      for (const key in features) {
        payload[key] = parseFloat(features[key])
        if (isNaN(payload[key])) throw new Error(`Invalid value for ${key.replace('_', ' ')}`)
      }

      // We use localhost:8000 for local dev
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || 'Prediction failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>OncoVision AI</h1>
        <p>Advanced SVM-based Breast Cancer Classification</p>
      </header>

      <main className="main-content">
        <section className="glass-panel form-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 600 }}>Tumor Features</h2>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button type="button" className="btn btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '0.875rem' }} onClick={() => handleRandomSample('benign')}>Load Benign</button>
              <button type="button" className="btn btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '0.875rem' }} onClick={() => handleRandomSample('malignant')}>Load Malignant</button>
            </div>
          </div>
          
          <form onSubmit={handleSubmit}>
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
                    onChange={handleChange}
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
                disabled={loading}
              >
                Clear
              </button>
              <button type="submit" className="btn btn-primary" disabled={loading}>
                {loading ? <span className="spinner"></span> : 'Run Prediction'}
              </button>
            </div>
          </form>
          {error && <div style={{ color: 'var(--danger-color)', marginTop: '1rem', padding: '1rem', background: 'rgba(239,68,68,0.1)', borderRadius: '8px' }}>{error}</div>}
        </section>

        <section className="glass-panel result-panel">
          {!result ? (
            <div className="result-placeholder">
              <div className="icon-placeholder">🔬</div>
              <h3>Awaiting Data</h3>
              <p style={{ textAlign: 'center' }}>Enter tumor features or load a sample to see the AI prediction.</p>
            </div>
          ) : (
            <div className="prediction-result">
              <h2 style={{ fontSize: '1.25rem', color: 'var(--text-muted)', marginBottom: '1rem', textAlign: 'center' }}>AI Diagnosis</h2>
              <div style={{ textAlign: 'center' }}>
                <div className={`status-badge ${result.prediction === 'Malignant' ? 'status-malignant' : 'status-benign'}`}>
                  {result.prediction}
                </div>
              </div>
              
              <div style={{ marginTop: '2rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Confidence</span>
                  <span className="confidence-text">{result.confidence}</span>
                </div>
                
                <div className="confidence-bar-container">
                  <div 
                    className={`confidence-bar ${result.prediction === 'Malignant' ? 'bar-malignant' : 'bar-benign'}`}
                    style={{ width: result.confidence }}
                  ></div>
                </div>
              </div>

              <div className="prob-details">
                <div className="prob-item">
                  <span className="prob-label">Benign Prob.</span>
                  <span className="prob-value" style={{ color: 'var(--success-color)' }}>{result.probabilities.Benign}</span>
                </div>
                <div className="prob-item" style={{ textAlign: 'right' }}>
                  <span className="prob-label">Malignant Prob.</span>
                  <span className="prob-value" style={{ color: 'var(--danger-color)' }}>{result.probabilities.Malignant}</span>
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
