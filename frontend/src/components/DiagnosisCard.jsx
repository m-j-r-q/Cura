function DiagnosisCard({ diagnosis }) {
  const { disease, confidence, affected_region, heatmap_base64 } = diagnosis

  return (
    <div className="diagnosis-card">
      <div className="diagnosis-header">
        <h3 className="disease-name">{disease}</h3>
        <span className="region-badge">{affected_region}</span>
      </div>

      <div className="confidence-row">
        <span className="confidence-label">Confidence</span>
        <span className="confidence-value">{(confidence * 100).toFixed(1)}%</span>
      </div>
      <div className="confidence-bar-track">
        <div
          className="confidence-bar-fill"
          style={{ width: `${confidence * 100}%` }}
        />
      </div>

      {heatmap_base64 && (
        <div className="heatmap-container">
          <p className="heatmap-label">Grad-CAM Activation Map</p>
          <img
            src={`data:image/png;base64,${heatmap_base64}`}
            alt={`Grad-CAM for ${disease}`}
            className="heatmap-image"
          />
        </div>
      )}
    </div>
  )
}

export default DiagnosisCard