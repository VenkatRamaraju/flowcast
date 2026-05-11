import { motion } from "framer-motion";
import { Bike, Github } from "lucide-react";

interface Props {
  stationCount: number;
  loading: boolean;
}

export function HeaderBar({ stationCount, loading }: Props) {
  return (
    <motion.header
      initial={{ y: -16, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-start p-4 md:p-6"
    >
      <div className="glass pointer-events-auto flex items-center gap-3 rounded-2xl px-3.5 py-2.5 shadow-glass">
        <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-flow-inflow/30 to-flow-inflow/5 ring-1 ring-flow-inflow/30">
          <Bike className="h-4 w-4 text-flow-inflow" strokeWidth={2.2} />
        </div>
        <div className="leading-tight">
          <div className="text-[13px] font-semibold tracking-tight text-white">
            Flowcast
          </div>
          <div className="text-[10.5px] uppercase tracking-[0.18em] text-white/40">
            Bay Area · Live
          </div>
        </div>
        <div className="mx-2 h-7 w-px bg-white/8" />
        <LiveDot loading={loading} />
        <div className="text-[11.5px] tabular text-white/55">
          {loading ? (
            <span className="inline-block w-12 skeleton h-3" />
          ) : (
            <>
              <span className="text-white/85 font-medium">{stationCount.toLocaleString()}</span>{" "}
              stations
            </>
          )}
        </div>
      </div>
      <div className="ml-auto pointer-events-auto">
        <div className="glass flex items-center gap-3 rounded-2xl px-3.5 py-2.5 shadow-glass">
          <a
            href="https://github.com/VenkatRamaraju/flowcast#flowcast"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 text-[11.5px] text-white/80 transition hover:text-white"
          >
            <Github className="h-3.5 w-3.5" />
            <span className="font-medium">README</span>
          </a>
          <div className="h-3.5 w-px bg-white/10" />
          <a
            href="https://venkatramaraju.github.io/website/"
            target="_blank"
            rel="noreferrer"
            className="text-[11.5px] text-white/70 transition hover:text-white"
          >
            Created by <span className="text-white/90">Venkat Ramaraju</span>
          </a>
        </div>
      </div>
    </motion.header>
  );
}

function LiveDot({ loading }: { loading: boolean }) {
  return (
    <span className="relative flex h-2 w-2 items-center justify-center" aria-hidden>
      {!loading && (
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-flow-inflow opacity-60" />
      )}
      <span
        className={`relative inline-flex h-2 w-2 rounded-full ${loading ? "bg-white/40" : "bg-flow-inflow"}`}
      />
    </span>
  );
}
