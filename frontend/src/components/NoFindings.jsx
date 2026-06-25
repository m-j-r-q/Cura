function NoFindings({ onReset }) {
  return (
    <div className="no-findings">
      <div className="no-findings-icon">✓</div>
      <h2>No Findings Detected</h2>
      <p>The model did not detect any pathology above the confidence threshold.</p>
      <button className="reset-btn" onClick={onReset}>
        Analyze Another Image
      </button>
    </div>
  )
}

export default NoFindings