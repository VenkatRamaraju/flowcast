import { motion } from "framer-motion";

interface Props {
  message?: string;
  isError?: boolean;
}

export function BootSplash({ message = "Connecting to Flowcast…", isError }: Props) {
  return (
    <motion.div
      className="absolute inset-0 z-30 flex items-center justify-center bg-ink-950"
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex flex-col items-center">
        <div className="relative h-14 w-14">
          <motion.div
            className="absolute inset-0 rounded-full border border-flow-inflow/30"
            animate={{ scale: [1, 1.5, 1.5], opacity: [0.7, 0, 0] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: "easeOut" }}
          />
          <motion.div
            className="absolute inset-0 rounded-full border border-flow-inflow/30"
            animate={{ scale: [1, 1.5, 1.5], opacity: [0.7, 0, 0] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: "easeOut", delay: 0.6 }}
          />
          <div className="absolute inset-3 rounded-full bg-flow-inflow shadow-[0_0_24px_rgba(52,227,194,0.7)]" />
        </div>
        <div className="mt-6 text-[12px] uppercase tracking-[0.32em] text-white/45">
          {isError ? "Connection failed" : message}
        </div>
        {isError && (
          <div className="mt-2 max-w-xs text-center text-[11.5px] text-white/35">
            Make sure the backend is running:&nbsp;
            <code className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-white/70">
              python main.py --server
            </code>
          </div>
        )}
      </div>
    </motion.div>
  );
}
