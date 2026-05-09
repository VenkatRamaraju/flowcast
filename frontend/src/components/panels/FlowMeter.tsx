import { motion } from "framer-motion";
import { useMemo } from "react";

import type { FlowDirection } from "@/types";
import { FLOW_COLORS } from "@/lib/colors";

interface Props {
  /** Predicted net flow for the station, signed. */
  value: number;
  /** Saturation point of the meter. Values above this clip to 100% width. */
  scale?: number;
  direction: FlowDirection;
}

/**
 * A center-anchored bar: outflow extends left, inflow extends right.
 * Fills with a spring on mount and animates when the value updates.
 */
export function FlowMeter({ value, scale = 12, direction }: Props) {
  const pct = useMemo(() => {
    const clipped = Math.max(-scale, Math.min(scale, value));
    return (clipped / scale) * 50;
  }, [value, scale]);

  const c = FLOW_COLORS[direction];

  return (
    <div className="relative">
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-white/[0.04]">
        {/* Subtle vertical center tick */}
        <div className="absolute left-1/2 top-0 h-full w-px bg-white/10" />

        {/* Animated fill */}
        <motion.div
          className="absolute top-0 h-full rounded-full"
          style={{
            backgroundColor: c.fg,
            boxShadow: `0 0 18px ${c.fg}55`,
          }}
          initial={{ width: 0, left: "50%" }}
          animate={{
            width: `${Math.abs(pct)}%`,
            left: pct >= 0 ? "50%" : `${50 + pct}%`,
          }}
          transition={{ type: "spring", stiffness: 180, damping: 24 }}
        />
      </div>

      {/* Tick labels */}
      <div className="mt-2 flex justify-between text-[10px] uppercase tracking-[0.18em] text-white/30">
        <span>−{scale} out</span>
        <span>0</span>
        <span>+{scale} in</span>
      </div>
    </div>
  );
}
