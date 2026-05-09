import { apiGet } from "./client";
import type { LiveResponse, Station, StationMapping } from "@/types";

export async function fetchStations(): Promise<Station[]> {
  const mapping = await apiGet<StationMapping>("/stations");
  const out: Station[] = [];
  for (const [id, coords] of Object.entries(mapping)) {
    if (!Array.isArray(coords) || coords.length < 2) continue;
    const [lat, lon] = coords;
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    out.push({
      id,
      region: id.split(/[-\s]/)[0] ?? "",
      lat,
      lon,
    });
  }
  return out;
}

export function fetchStationLive(stationId: string): Promise<LiveResponse> {
  return apiGet<LiveResponse>(`/stations/${encodeURIComponent(stationId)}/live`);
}
