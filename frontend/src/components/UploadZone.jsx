import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

function UploadZone({ onFileSelected }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onFileSelected(acceptedFiles[0])
    }
  }, [onFileSelected])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg'] },
    multiple: false,
  })

  return (
    <div className="upload-wrapper">
      <h2 className="upload-title">Upload Chest X-Ray</h2>
      <p className="upload-subtitle">
        Supports PNG and JPEG. Image will be analyzed for 14 pathologies.
      </p>

      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'dropzone-active' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="dropzone-icon">⊕</div>
        {isDragActive
          ? <p>Drop the image here</p>
          : <p>Drag and drop an X-ray here, or click to browse</p>
        }
        <span className="dropzone-hint">PNG, JPG up to 10MB</span>
      </div>
    </div>
  )
}

export default UploadZone