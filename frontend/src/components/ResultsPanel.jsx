import DiagnosisCard from './DiagnosisCard'
import NoFindings from './NoFindings'

function QualityScoreBar({ label, score }) {
  const color = score >= 75 ? '#22c55e' 
              : score >= 50 ? '#f59e0b' 
              : '#ef4444'

  return (
    <div className="quality-score-row">
      <div className="quality-score-header">
        <span className="quality-score-label">{label}</span>
        <span className="quality-score-value" style={{ color }}>
          {score}%
        </span>
      </div>
      <div className="confidence-bar-track">
        <div
          className="confidence-bar-fill"
          style={{ width: `${score}%`, background: color }}
        />
      </div>
    </div>
  )
}


function ResultsPanel({ report, onReset }) {
  const { image_quality, quality_metrics, diagnoses, models } = report

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
        <span className="model-label">Model: {models}</span>
      </div>

      {report.quality_scores && (
        <div className="quality-scores-panel">
          <p className="quality-overall">
            Overall image quality: 
            <strong> {report.quality_scores.overall}%</strong>
          </p>
          <QualityScoreBar label="Sharpness" score={report.quality_scores.sharpness} />
          <QualityScoreBar label="Contrast"  score={report.quality_scores.contrast} />
          <QualityScoreBar label="Exposure"  score={report.quality_scores.exposure} />
        </div>
      )}

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