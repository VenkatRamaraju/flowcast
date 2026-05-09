import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowDownRight,
  ArrowUpRight,
  Minus,
  RefreshCw,
  X,
  MapPin,
  AlertCircle,
} from "lucide-react";

import type { Station } from "@/types";
import { useStationLive } from "@/hooks/useStations";
import { useAppStore } from "@/store/useAppStore";
import { classifyFlow, magnitudeLabel, regionLabel } from "@/lib/format";
import { FLOW_COLORS } from "@/lib/colors";
import { AnimatedNumber } from "@/components/ui/AnimatedNumber";
import { FlowMeter } from "./FlowMeter";

interface Props {
  station: Station | null;
}

export function StationDetailPanel({ station }: Props) {
  return (
    <AnimatePresence mode="wait">
      {station && <PanelInner key={station.id} station={station} />}
    </AnimatePresence>
  );
}

function PanelInner({ station }: { station: Station }) {
  const { data, isLoading, isError, error, refetch, isFetching, dataUpdatedAt } =
    useStationLive(station.id);
  const close = useAppStore((s) => s.selectStation);

  const direction = data ? classifyFlow(data.prediction) : "balanced";
  const colors = FLOW_COLORS[direction];

  return (
    <motion.aside
      initial={{ x: 32, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 32, opacity: 0 }}
      transition={{ type: "spring", stiffness: 280, damping: 30, mass: 0.9 }}
      className="glass-strong pointer-events-auto absolute right-4 top-20 z-20 w-[400px] max-w-[calc(100vw-2rem)] overflow-hidden rounded-3xl shadow-panel md:right-6 md:top-24"
      role="dialog"
      aria-label={`Station ${station.id} details`}
    >
      {/* Color accent stripe along the top, tinted by flow direction. */}
      <motion.div
        className="absolute inset-x-0 top-0 h-px"
        style={{
          background: `linear-gradient(90deg, transparent, ${colors.fg}, transparent)`,
        }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.15 }}
      />

      <header className="flex items-start justify-between px-6 pt-5">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-white/40">
            <MapPin className="h-3 w-3" strokeWidth={2.4} />
            {regionLabel(station.region)}
          </div>
          <h2 className="mt-1.5 truncate font-mono text-[18px] font-medium tracking-tight text-white">
            {station.id}
          </h2>
          <p className="tabular mt-0.5 text-[11.5px] text-white/40">
            {station.lat.toFixed(4)}°N · {Math.abs(station.lon).toFixed(4)}°W
          </p>
        </div>
        <button
          onClick={() => close(null)}
          className="-mr-1.5 -mt-1 rounded-xl p-2 text-white/40 transition-colors hover:bg-white/[0.06] hover:text-white"
          aria-label="Close"
        >
          <X className="h-4 w-4" strokeWidth={2.2} />
        </button>
      </header>

      <div className="mt-5 px-6 pb-6">
        <div className="text-[10.5px] uppercase tracking-[0.22em] text-white/40">
          Predicted net bike flow
        </div>
        <div className="mt-1 text-[10.5px] tabular text-white/30">
          15-minute window · live
        </div>

        {/* Main metric area: skeleton, error, or value */}
        <div className="mt-4 min-h-[88px]">
          {isLoading ? (
            <SkeletonMetric />
          ) : isError ? (
            <ErrorState
              message={error instanceof Error ? error.message : "Failed to load"}
              onRetry={() => refetch()}
            />
          ) : data ? (
            <Metric value={data.prediction} direction={direction} />
          ) : null}
        </div>

        {/* Flow meter: bar visualization */}
        <div className="mt-6">
          {isLoading || !data ? (
            <div className="space-y-2">
              <div className="skeleton h-2 w-full" />
              <div className="skeleton h-2 w-1/3" />
            </div>
          ) : (
            <FlowMeter value={data.prediction} direction={direction} />
          )}
        </div>

        {/* Secondary stats grid */}
        <div className="mt-6 grid grid-cols-2 gap-2.5">
          <Stat
            label="Direction"
            value={
              isLoading || !data
                ? null
                : direction === "inflow"
                  ? "Bikes arriving"
                  : direction === "outflow"
                    ? "Bikes departing"
                    : "Equilibrium"
            }
            color={colors.fg}
          />
          <Stat
            label="Magnitude"
            value={isLoading || !data ? null : magnitudeLabel(data.prediction)}
            color="#cfd6e6"
          />
        </div>

        {/* Footer: refresh + last updated */}
        <div className="mt-6 flex items-center justify-between border-t hairline pt-3">
          <span className="text-[10.5px] uppercase tracking-[0.18em] text-white/35">
            {dataUpdatedAt ? `Updated ${formatRelative(dataUpdatedAt)}` : "—"}
          </span>
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="group flex items-center gap-1.5 rounded-lg px-2 py-1 text-[11.5px] text-white/55 transition-colors hover:bg-white/[0.06] hover:text-white disabled:opacity-50"
          >
            <RefreshCw
              className={`h-3 w-3 transition-transform ${isFetching ? "animate-spin" : "group-hover:rotate-90"}`}
              strokeWidth={2.2}
            />
            Refresh
          </button>
        </div>
      </div>
    </motion.aside>
  );
}

function Metric({
  value,
  direction,
}: {
  value: number;
  direction: ReturnType<typeof classifyFlow>;
}) {
  const c = FLOW_COLORS[direction];
  const Icon =
    direction === "inflow" ? ArrowDownRight : direction === "outflow" ? ArrowUpRight : Minus;

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
      className="flex items-end gap-3"
    >
      <div className="leading-none">
        <span
          className="font-sans text-[56px] font-medium tracking-[-0.04em]"
          style={{ color: c.fg, textShadow: `0 0 32px ${c.fg}33` }}
        >
          <AnimatedNumber value={value} signed decimals={1} duration={650} />
        </span>
        <span className="ml-1 text-[14px] text-white/40">bikes / 15 min</span>
      </div>
      <div
        className="mb-2 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10.5px] font-medium uppercase tracking-[0.16em]"
        style={{
          backgroundColor: c.bg,
          color: c.fg,
          boxShadow: `0 0 0 1px ${c.ring} inset`,
        }}
      >
        <Icon className="h-3 w-3" strokeWidth={2.6} />
        {direction}
      </div>
    </motion.div>
  );
}

function SkeletonMetric() {
  return (
    <div className="space-y-3">
      <div className="flex items-end gap-3">
        <div className="skeleton h-12 w-32" />
        <div className="skeleton mb-2 h-5 w-20 rounded-full" />
      </div>
      <div className="skeleton h-3 w-40" />
    </div>
  );
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border hairline bg-white/[0.02] p-4">
      <AlertCircle className="mt-0.5 h-4 w-4 text-flow-outflow" strokeWidth={2.2} />
      <div className="min-w-0 flex-1">
        <div className="text-[12.5px] font-medium text-white/85">
          Couldn't reach forecaster
        </div>
        <div className="mt-0.5 truncate text-[11.5px] text-white/50">{message}</div>
        <button
          onClick={onRetry}
          className="mt-2 inline-flex items-center gap-1.5 rounded-lg bg-white/[0.06] px-2 py-1 text-[11.5px] text-white/85 transition-colors hover:bg-white/[0.12]"
        >
          <RefreshCw className="h-3 w-3" strokeWidth={2.2} />
          Try again
        </button>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  color,
}: {
  label: string;
  value: string | null;
  color: string;
}) {
  return (
    <div className="rounded-xl border hairline bg-white/[0.015] px-3 py-2.5">
      <div className="text-[10px] uppercase tracking-[0.18em] text-white/40">{label}</div>
      {value ? (
        <div className="mt-1 text-[13px] font-medium capitalize" style={{ color }}>
          {value}
        </div>
      ) : (
        <div className="skeleton mt-1.5 h-3.5 w-20" />
      )}
    </div>
  );
}

function formatRelative(ts: number): string {
  const diff = Math.max(0, Date.now() - ts);
  if (diff < 5_000) return "just now";
  if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`;
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  return new Date(ts).toLocaleTimeString();
}
