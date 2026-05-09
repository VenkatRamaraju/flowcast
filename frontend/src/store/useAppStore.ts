import { create } from "zustand";
import type { StationId } from "@/types";

interface AppState {
  selectedStationId: StationId | null;
  hoveredStationId: StationId | null;
  selectStation: (id: StationId | null) => void;
  setHovered: (id: StationId | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  selectedStationId: null,
  hoveredStationId: null,
  selectStation: (id) => set({ selectedStationId: id }),
  setHovered: (id) => set({ hoveredStationId: id }),
}));
