import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, FileText, Gauge, Footprints, Navigation } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Navbar from "@/components/Navbar";

interface Recommendation {
  recommendation: string;
  verdict: string;
  arc_start?: number;
  arc_end?: number;
  mean_delta?: number;
  offset_m?: number;
  category: "throttle" | "brake" | "steering" | "other";
}

interface ResultsData {
  recommendations: Recommendation[];
}

const categoryConfig = {
  throttle: { label: "Throttle", icon: Gauge, color: "text-green-400", border: "border-green-500/30" },
  brake: { label: "Brake", icon: Footprints, color: "text-red-400", border: "border-red-500/30" },
  steering: { label: "Steering", icon: Navigation, color: "text-blue-400", border: "border-blue-500/30" },
  other: { label: "Other", icon: FileText, color: "text-muted-foreground", border: "border-border" },
};

const Results = () => {
  const navigate = useNavigate();
  const [data, setData] = useState<ResultsData | null>(null);

  useEffect(() => {
    const raw = sessionStorage.getItem("lapResults");
    if (raw) {
      try { setData(JSON.parse(raw)); } catch { /* ignore */ }
    }
  }, []);

  if (!data) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <div className="pt-32 px-6 text-center">
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-md mx-auto">
            <div className="w-16 h-16 rounded-2xl gradient-glow flex items-center justify-center mx-auto mb-6">
              <FileText className="w-7 h-7 text-primary" />
            </div>
            <h2 className="text-2xl font-bold mb-3">No results yet</h2>
            <p className="text-muted-foreground mb-8">Upload your MCAP files on the home page to see analysis here.</p>
            <Button onClick={() => navigate("/")} variant="outline" className="gap-2">
              <ArrowLeft className="w-4 h-4" /> Back to Upload
            </Button>
          </motion.div>
        </div>
      </div>
    );
  }

  const { recommendations } = data;
  const grouped = {
    throttle: recommendations.filter((r) => r.category === "throttle"),
    brake: recommendations.filter((r) => r.category === "brake"),
    steering: recommendations.filter((r) => r.category === "steering"),
    other: recommendations.filter((r) => r.category === "other"),
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <div className="pt-28 pb-16 px-6">
        <div className="max-w-4xl mx-auto">
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
            <Button onClick={() => navigate("/")} variant="ghost" size="sm" className="gap-2 mb-6 text-muted-foreground">
              <ArrowLeft className="w-4 h-4" /> New Analysis
            </Button>
            <h1 className="text-3xl font-bold mb-2">Coaching Recommendations</h1>
            <p className="text-muted-foreground mb-8">
              {recommendations.length} suggestions to improve your lap time.
            </p>
          </motion.div>

          {(["throttle", "brake", "steering", "other"] as const).map((cat) => {
            const items = grouped[cat];
            if (items.length === 0) return null;
            const cfg = categoryConfig[cat];
            const Icon = cfg.icon;

            return (
              <motion.div key={cat} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
                <div className="flex items-center gap-2 mb-4">
                  <Icon className={`w-5 h-5 ${cfg.color}`} />
                  <h2 className="text-xl font-semibold">{cfg.label}</h2>
                  <span className="text-xs text-muted-foreground ml-1">({items.length})</span>
                </div>
                <div className="space-y-3">
                  {items.map((rec, i) => (
                    <Card key={i} className={`bg-card ${cfg.border} border`}>
                      <CardContent className="p-4">
                        <p className="text-sm text-foreground leading-relaxed">{rec.recommendation}</p>
                        <div className="flex flex-wrap gap-3 mt-2 text-xs text-muted-foreground font-mono">
                          {rec.arc_start != null && rec.arc_end != null && (
                            <span>{rec.arc_start} m – {rec.arc_end} m</span>
                          )}
                          {rec.offset_m != null && <span>offset: {rec.offset_m} m</span>}
                          {rec.mean_delta != null && <span>Δ: {rec.mean_delta}</span>}
                          <span className="opacity-60">{rec.verdict}</span>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Results;
