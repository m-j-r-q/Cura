import { useState } from 'react'
import axios from 'axios'

import Header from './components/Header'
import UploadZone from './components/UploadZone'
import LoadingScreen from './components/LoadingScreen'
import QualityRejected from './components/QualityRejected'
import ResultsPanel from './components/ResultsPanel'

const API_URL = import.meta.env.VITE_API_URL || ''

function App() {
  const [state, setState] = useState('idle')
  const [report, setReport] = useState(null)
  const [error, setError]   = useState(null)

  const handleFileSelected = async (file) => {
    setState('loading')
    setReport(null)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post(
        API_URL ? `${API_URL}/analyze` : '/api/analyze',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )

      const data = response.data

      if (!data.passed_quality) {
        setState('rejected')
      } else {
        setState('results')
      }

      setReport(data)

    } catch (err) {
      setError(err.message)
      setState('idle')
    }
  }

  const handleReset = () => {
    setState('idle')
    setReport(null)
    setError(null)
  }

  return (
    <div className="app">
      <Header />

      <main className="main">
        {state === 'idle' && (
          <UploadZone onFileSelected={handleFileSelected} />
        )}

        {state === 'loading' && (
          <LoadingScreen />
        )}

        {state === 'rejected' && report && (
          <QualityRejected
            reason={report.rejection_reason}
            metrics={report.quality_metrics}
            onReset={handleReset}
          />
        )}

        {state === 'results' && report && (
          <ResultsPanel
            report={report}
            onReset={handleReset}
          />
        )}

        {error && (
          <p className="error-text">Error: {error}</p>
        )}
      </main>
    </div>
  )
}

export default App