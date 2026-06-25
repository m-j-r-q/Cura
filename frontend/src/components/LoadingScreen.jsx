function LoadingScreen() {
  return (
    <div className="loading-screen">
      <div className="spinner" />
      <p className="loading-text">Analyzing image...</p>
      <p className="loading-sub">Running quality assessment, inference and explainability pipeline</p>
    </div>
  )
}

export default LoadingScreen