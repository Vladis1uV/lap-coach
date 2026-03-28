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
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<UploadState>("idle");
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");

  const handleFile = useCallback((f: File) => {
    if (!f.name.endsWith(".mcap")) {
      setError("Please upload a .mcap file");
      setState("error");
      return;
    }
    setFile(f);
    setState("selected");
    setError("");
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    },
    [handleFile]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) handleFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    setState("uploading");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${apiUrl}/api/process`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      onFileProcessed(data.result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setState("error");
    }
  };

  return (
    <div className="w-full max-w-xl mx-auto">
      <motion.div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={cn(
          "relative rounded-2xl border-2 border-dashed p-10 text-center transition-colors cursor-pointer",
          dragOver
            ? "border-primary bg-primary/5"
            : state === "error"
            ? "border-destructive/50 bg-destructive/5"
            : state === "selected"
            ? "border-primary/50 bg-primary/5"
            : "border-border hover:border-muted-foreground/40"
        )}
        whileHover={{ scale: 1.01 }}
        onClick={() => {
          if (state !== "uploading") document.getElementById("mcap-input")?.click();
        }}
      >
        <input
          id="mcap-input"
          type="file"
          accept=".mcap"
          className="hidden"
          onChange={onInputChange}
        />

        <AnimatePresence mode="wait">
          {state === "idle" && (
            <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <div className="w-16 h-16 rounded-2xl gradient-glow flex items-center justify-center">
                <Upload className="w-7 h-7 text-primary" />
              </div>
              <p className="text-lg font-medium text-foreground">Drop your MCAP file here</p>
              <p className="text-sm text-muted-foreground">or click to browse · .mcap files only</p>
            </motion.div>
          )}

          {state === "selected" && file && (
            <motion.div key="selected" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center">
                <FileCheck className="w-7 h-7 text-primary" />
              </div>
              <p className="text-lg font-medium text-foreground">{file.name}</p>
              <p className="text-sm text-muted-foreground">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
            </motion.div>
          )}

          {state === "uploading" && (
            <motion.div key="uploading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <Loader2 className="w-10 h-10 text-primary animate-spin" />
              <p className="text-lg font-medium text-foreground">Processing...</p>
              <p className="text-sm text-muted-foreground">Analyzing your lap data</p>
            </motion.div>
          )}

          {state === "error" && (
            <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <div className="w-16 h-16 rounded-2xl bg-destructive/10 flex items-center justify-center">
                <AlertCircle className="w-7 h-7 text-destructive" />
              </div>
              <p className="text-lg font-medium text-foreground">Something went wrong</p>
              <p className="text-sm text-destructive">{error}</p>
              <p className="text-xs text-muted-foreground mt-1">Click to try again</p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {state === "selected" && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mt-6 flex justify-center">
          <Button size="lg" onClick={(e) => { e.stopPropagation(); handleUpload(); }} className="px-8 font-semibold text-base">
            Analyze Lap Data
          </Button>
        </motion.div>
      )}
    </div>
  );
};

export default FileUpload;
