import DiagnosisCard from './DiagnosisCard'
import NoFindings from './NoFindings'

function ResultsPanel({ report, onReset }) {
  const { image_quality, quality_metrics, diagnoses, model } = report

  const qualityColor = {
    'Good':       '#22c55e',
    'Acceptable': '#f59e0b',
    'Poor':       '#ef4444',
  }

  return (
    <div className="results-panel">
      <div className="results-header">
        <div className="quality-badge" style={{ borderColor: qualityColor[image_quality] }}>
          <span style={{ color: qualityColor[image_quality] }}>
            Image Quality: {image_quality}
          </span>
        </div>
        <span className="model-label">Model: {model}</span>
      </div>

      <div className="metrics-grid">
        <div className="metric-box">
          <span className="metric-label">Blur Score</span>
          <span className="metric-value">{quality_metrics.blur_score}</span>
        </div>
        <div className="metric-box">
          <span className="metric-label">Contrast</span>
          <span className="metric-value">{quality_metrics.contrast_score}</span>
        </div>
        <div className="metric-box">
          <span className="metric-label">Brightness</span>
          <span className="metric-value">{quality_metrics.mean_brightness}</span>
        </div>
      </div>

      <h2 className="findings-title">
        {diagnoses.length > 0 ? `${diagnoses.length} Findings Detected` : 'No Findings'}
      </h2>

      {diagnoses.length === 0 ? (
        <NoFindings onReset={onReset} />
      ) : (
        <div className="diagnosis-grid">
          {diagnoses.map((d, idx) => (
            <DiagnosisCard key={idx} diagnosis={d} />
          ))}
        </div>
      )}

      <button className="reset-btn" onClick={onReset}>
        Analyze Another Image
      </button>
    </div>
  )
}

export default ResultsPanel