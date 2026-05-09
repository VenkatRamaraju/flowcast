import { useEffect, useMemo } from "react";
import { AnimatePresence } from "framer-motion";

import { useStations } from "@/hooks/useStations";
import { useAppStore } from "@/store/useAppStore";
import { FlowcastMap } from "@/components/map/FlowcastMap";
import { HeaderBar } from "@/components/panels/HeaderBar";
import { Legend } from "@/components/panels/Legend";
import { StationDetailPanel } from "@/components/panels/StationDetailPanel";
import { BootSplash } from "@/components/panels/BootSplash";

export default function App() {
  const { data: stations, isLoading, isError } = useStations();
  const selectedId = useAppStore((s) => s.selectedStationId);
  const selectStation = useAppStore((s) => s.selectStation);

  const selectedStation = useMemo(() => {
    if (!stations || !selectedId) return null;
    return stations.find((s) => s.id === selectedId) ?? null;
  }, [stations, selectedId]);

  // Keyboard shortcuts.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && selectedId) {
        selectStation(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedId, selectStation]);

  const showSplash = isLoading || isError;

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-ink-950">
      <FlowcastMap stations={stations ?? []} />
      <HeaderBar stationCount={stations?.length ?? 0} loading={isLoading} />
      <Legend />
      <StationDetailPanel station={selectedStation} />

      <AnimatePresence>{showSplash && <BootSplash isError={isError} />}</AnimatePresence>
    </div>
  );
}
