import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Gauge, Zap, BarChart3 } from "lucide-react";
import FileUpload from "@/components/FileUpload";
import Navbar from "@/components/Navbar";

const features = [
  { icon: Gauge, title: "Lap Analysis", desc: "Break down every corner, straight, and braking zone with precision telemetry." },
  { icon: Zap, title: "Performance Tips", desc: "Get actionable coaching suggestions to shave seconds off your lap time." },
  { icon: BarChart3, title: "Visual Insights", desc: "Interactive charts and overlays to compare your runs side by side." },
];

const Index = () => {
  const navigate = useNavigate();

  const handleProcessed = (data: unknown) => {
    const withUrl = { ...(data as Record<string, unknown>), apiUrl: "http://localhost:8000" };
    sessionStorage.setItem("lapResults", JSON.stringify(withUrl));
    navigate("/results");
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      {/* Hero */}
      <section className="pt-32 pb-16 px-6">
        <div className="max-w-3xl mx-auto text-center">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
            <p className="text-sm font-mono font-medium tracking-widest text-primary uppercase mb-4">
              Autonomous Racing Telemetry
            </p>
            <h1 className="text-5xl md:text-6xl font-bold tracking-tight leading-[1.1] mb-6">
              Your <span className="text-gradient">Lap Coach</span>,{" "}
              <br className="hidden md:block" />
              powered by data.
            </h1>
            <p className="text-lg text-muted-foreground max-w-xl mx-auto">
              {/* TODO: Add your project description here */}
              Upload your MCAP telemetry file and get instant analysis of your autonomous racing performance — corner speeds, braking zones, and coaching tips.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Upload */}
      <section className="pb-20 px-6">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3, duration: 0.6 }}>
          <FileUpload onFileProcessed={handleProcessed} />
        </motion.div>
      </section>

      {/* Features */}
      <section className="pb-24 px-6">
        <div className="max-w-4xl mx-auto grid md:grid-cols-3 gap-6">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 + i * 0.1, duration: 0.5 }}
              className="rounded-2xl border border-border bg-card p-6 hover:border-primary/30 transition-colors"
            >
              <div className="w-10 h-10 rounded-xl gradient-glow flex items-center justify-center mb-4">
                <f.icon className="w-5 h-5 text-primary" />
              </div>
              <h3 className="font-semibold text-lg mb-2">{f.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8 px-6 text-center text-xs text-muted-foreground">
        Lap Coach · Built for speed
      </footer>
    </div>
  );
};

export default Index;
