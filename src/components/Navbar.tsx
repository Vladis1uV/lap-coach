import { NavLink } from "@/components/NavLink";
import { Activity } from "lucide-react";

const Navbar = () => (
  <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl">
    <div className="max-w-6xl mx-auto flex items-center justify-between px-6 h-16">
      <NavLink to="/" className="flex items-center gap-2.5 group">
        <div className="w-8 h-8 rounded-lg bg-primary/15 flex items-center justify-center group-hover:bg-primary/25 transition-colors">
          <Activity className="w-4 h-4 text-primary" />
        </div>
        <span className="font-semibold text-lg tracking-tight">Lap Coach</span>
      </NavLink>
      <div className="flex items-center gap-6 text-sm">
        <NavLink to="/" className="text-muted-foreground hover:text-foreground transition-colors" activeClassName="text-foreground">
          Home
        </NavLink>
        <NavLink to="/results" className="text-muted-foreground hover:text-foreground transition-colors" activeClassName="text-foreground">
          Results
        </NavLink>
      </div>
    </div>
  </nav>
);

export default Navbar;
