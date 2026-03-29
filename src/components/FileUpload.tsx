import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileCheck, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface FileUploadProps {
  onFileProcessed: (data: unknown) => void;
  apiUrl?: string;
}

type UploadState = "idle" | "selected" | "uploading" | "error";

const FileUpload = ({ onFileProcessed, apiUrl = "http://localhost:8000" }: FileUploadProps) => {
  const [fastFile, setFastFile] = useState<File | null>(null);
  const [goodFile, setGoodFile] = useState<File | null>(null);
  const [state, setState] = useState<UploadState>("idle");
  const [dragOver, setDragOver] = useState<"fast" | "good" | null>(null);
  const [error, setError] = useState("");

  const validateFile = (f: File): boolean => {
    if (!f.name.endsWith(".mcap")) {
      setError("Please upload a .mcap file");
      setState("error");
      return false;
    }
    setError("");
    return true;
  };

  const handleFastFile = useCallback((f: File) => {
    if (!validateFile(f)) return;
    setFastFile(f);
    if (state === "error") setState("idle");
  }, [state]);

  const handleGoodFile = useCallback((f: File) => {
    if (!validateFile(f)) return;
    setGoodFile(f);
    if (state === "error") setState("idle");
  }, [state]);

  const bothSelected = fastFile && goodFile;

  const handleUpload = async () => {
    if (!fastFile || !goodFile) return;
    setState("uploading");
    try {
      const formData = new FormData();
      formData.append("file_fast", fastFile);
      formData.append("file_good", goodFile);
      const res = await fetch(`${apiUrl}/api/process`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      onFileProcessed(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setState("error");
    }
  };

  const DropZone = ({
    label,
    file,
    onFile,
    dragId,
  }: {
    label: string;
    file: File | null;
    onFile: (f: File) => void;
    dragId: "fast" | "good";
  }) => {
    const inputId = `mcap-input-${dragId}`;
    return (
      <motion.div
        onDragOver={(e) => { e.preventDefault(); setDragOver(dragId); }}
        onDragLeave={() => setDragOver(null)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(null);
          if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
        }}
        onClick={() => {
          if (state !== "uploading") document.getElementById(inputId)?.click();
        }}
        className={cn(
          "relative rounded-2xl border-2 border-dashed p-8 text-center transition-colors cursor-pointer flex-1 min-w-[240px]",
          dragOver === dragId
            ? "border-primary bg-primary/5"
            : file
            ? "border-primary/50 bg-primary/5"
            : "border-border hover:border-muted-foreground/40"
        )}
        whileHover={{ scale: 1.01 }}
      >
        <input
          id={inputId}
          type="file"
          accept=".mcap"
          className="hidden"
          onChange={(e) => { if (e.target.files?.[0]) onFile(e.target.files[0]); }}
        />
        {file ? (
          <div className="flex flex-col items-center gap-2">
            <FileCheck className="w-6 h-6 text-primary" />
            <p className="text-sm font-medium text-foreground truncate max-w-full">{file.name}</p>
            <p className="text-xs text-muted-foreground">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <Upload className="w-6 h-6 text-muted-foreground" />
            <p className="text-sm font-medium text-foreground">{label}</p>
            <p className="text-xs text-muted-foreground">Drop .mcap or click to browse</p>
          </div>
        )}
      </motion.div>
    );
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div className="flex flex-col sm:flex-row gap-4">
        <DropZone label="Fast (Reference) Lap" file={fastFile} onFile={handleFastFile} dragId="fast" />
        <DropZone label="Good Lap (to tune)" file={goodFile} onFile={handleGoodFile} dragId="good" />
      </div>

      <AnimatePresence>
        {state === "uploading" && (
          <motion.div key="uploading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mt-6 flex flex-col items-center gap-2">
            <Loader2 className="w-8 h-8 text-primary animate-spin" />
            <p className="text-sm text-muted-foreground">Analyzing your laps...</p>
          </motion.div>
        )}
        {state === "error" && (
          <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mt-6 flex flex-col items-center gap-2">
            <AlertCircle className="w-6 h-6 text-destructive" />
            <p className="text-sm text-destructive">{error}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {bothSelected && state !== "uploading" && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mt-6 flex justify-center">
          <Button size="lg" onClick={handleUpload} className="px-8 font-semibold text-base">
            Analyze Lap Data
          </Button>
        </motion.div>
      )}
    </div>
  );
};

export default FileUpload;
