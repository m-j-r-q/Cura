function QualityRejected({ reason, metrics, onReset }) {
  return (
    <div className="rejected-panel">
      <div className="rejected-icon">✕</div>
      <h2 className="rejected-title">Image Rejected</h2>
      <p className="rejected-reason">{reason}</p>

      <div className="metrics-grid">
        <div className="metric-box">
          <span className="metric-label">Blur Score</span>
          <span className="metric-value">{metrics.blur_score}</span>
        </div>
        <div className="metric-box">
          <span className="metric-label">Contrast</span>
          <span className="metric-value">{metrics.contrast_score}</span>
        </div>
        <div className="metric-box">
          <span className="metric-label">Brightness</span>
          <span className="metric-value">{metrics.mean_brightness}</span>
        </div>
      </div>

      <button className="reset-btn" onClick={onReset}>
        Upload Another Image
      </button>
    </div>
  )
}

export default QualityRejected