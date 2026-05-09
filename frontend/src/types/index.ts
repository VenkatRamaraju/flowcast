export type StationId = string;

/** Raw `/stations` response: `{ station_id: [lat, lon] }`. */
export type StationMapping = Record<StationId, [number, number]>;

/** Normalized station record for UI consumption. */
export interface Station {
  id: StationId;
  /** Two-letter region/city code parsed from the id prefix (SF, SJ, OK, BK, EM, DC). */
  region: string;
  lat: number;
  lon: number;
}

/** Raw `/stations/{id}/live` response. */
export interface LiveResponse {
  station_id: StationId;
  prediction: number;
  temperature: number;
  precipitation: number;
  wind: number;
}

/** Direction of net flow for visual treatment. */
export type FlowDirection = "inflow" | "outflow" | "balanced";
