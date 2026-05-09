import { useQuery } from "@tanstack/react-query";
import { fetchStations, fetchStationLive } from "@/api/stations";

export function useStations() {
  return useQuery({
    queryKey: ["stations"],
    queryFn: fetchStations,
    staleTime: 5 * 60_000,
  });
}

export function useStationLive(stationId: string | null) {
  return useQuery({
    queryKey: ["station-live", stationId],
    queryFn: () => fetchStationLive(stationId!),
    enabled: !!stationId,
    staleTime: 30_000,
    refetchInterval: stationId ? 60_000 : false,
  });
}
