/**
 * Video Upload Component
 * 
 * Allows users to upload existing golf swing videos for analysis.
 */

import { useCallback, useRef, useState } from 'react';
import { Upload, Film, X } from 'lucide-react';

interface VideoUploadProps {
  onVideoSelected: (file: File, url: string) => void;
  accept?: string;
  maxSizeMB?: number;
}

export function VideoUpload({ 
  onVideoSelected, 
  accept = 'video/*',
  maxSizeMB = 100 
}: VideoUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    setError(null);

    // Validate file type
    if (!file.type.startsWith('video/')) {
      setError('Please select a video file');
      return;
    }

    // Validate file size
    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > maxSizeMB) {
      setError(`Video must be smaller than ${maxSizeMB}MB`);
      return;
    }

    // Create object URL for preview
    const url = URL.createObjectURL(file);
    onVideoSelected(file, url);
  }, [maxSizeMB, onVideoSelected]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) {
      handleFile(file);
    }
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
    }
  }, [handleFile]);

  const handleClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  return (
    <div className="w-full">
      <div
        onClick={handleClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          relative border-2 border-dashed rounded-2xl p-8
          flex flex-col items-center justify-center gap-4
          cursor-pointer transition-all duration-200
          ${isDragging 
            ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10' 
            : 'border-[var(--color-primary-light)] hover:border-[var(--color-accent)]/50 hover:bg-[var(--color-surface-card)]'
          }
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          onChange={handleInputChange}
          className="hidden"
          aria-label="Upload a golf swing video"
        />

        <div className={`
          w-16 h-16 rounded-full flex items-center justify-center
          ${isDragging ? 'bg-[var(--color-accent)]' : 'bg-[var(--color-surface-card)]'}
          transition-colors duration-200
        `}>
          {isDragging ? (
            <Film className="w-8 h-8 text-[var(--color-primary-dark)]" />
          ) : (
            <Upload className="w-8 h-8 text-[var(--color-accent)]" />
          )}
        </div>

        <div className="text-center">
          <p className="text-[var(--color-text)] font-medium">
            {isDragging ? 'Drop video here' : 'Upload a golf swing video'}
          </p>
          <p className="text-[var(--color-text-muted)] text-sm mt-1">
            Drag & drop or click to browse
          </p>
          <p className="text-[var(--color-text-muted)] text-xs mt-2">
            MP4, MOV, WebM • Max {maxSizeMB}MB
          </p>
        </div>
      </div>

      {error && (
        <div className="mt-3 flex items-center gap-2 text-red-400 text-sm">
          <X className="w-4 h-4" />
          {error}
        </div>
      )}
    </div>
  );
}