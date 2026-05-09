import type { FlowDirection } from "@/types";

/** Tailwind/CSS color tokens for flow states. Single source of truth. */
export const FLOW_COLORS: Record<FlowDirection, { fg: string; bg: string; ring: string }> = {
  inflow: {
    fg: "#34e3c2",
    bg: "rgba(52, 227, 194, 0.12)",
    ring: "rgba(52, 227, 194, 0.45)",
  },
  outflow: {
    fg: "#ff6f91",
    bg: "rgba(255, 111, 145, 0.12)",
    ring: "rgba(255, 111, 145, 0.45)",
  },
  balanced: {
    fg: "#9bb0d3",
    bg: "rgba(155, 176, 211, 0.10)",
    ring: "rgba(155, 176, 211, 0.35)",
  },
};
