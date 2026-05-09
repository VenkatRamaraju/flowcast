import type { FlowDirection } from "@/types";

/** Bay Area region labels, derived from the station id prefix. */
export const REGION_LABELS: Record<string, string> = {
  SF: "San Francisco",
  SJ: "San Jose",
  OK: "Oakland",
  BK: "Berkeley",
  EM: "Emeryville",
  DC: "Daly City",
};

export function regionLabel(code: string): string {
  return REGION_LABELS[code] ?? code;
}

export function classifyFlow(prediction: number): FlowDirection {
  if (prediction > 1) return "inflow";
  if (prediction < -1) return "outflow";
  return "balanced";
}

/** Compact, sign-aware number with a single decimal. */
export function formatFlow(value: number): string {
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${Math.abs(value).toFixed(1)}`;
}

export function magnitudeLabel(value: number): string {
  const m = Math.abs(value);
  if (m < 1) return "low";
  if (m < 4) return "moderate";
  if (m < 9) return "elevated";
  return "high";
}
