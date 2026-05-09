/**
 * Beautiful, free, attribution-friendly dark basemap from CARTO.
 * Vector tiles via MapLibre — GPU rasterized, ships without an API key.
 *
 * If you want to swap to Mapbox later, just point this at a `mapbox://` style
 * and instantiate Map with an `accessToken` field.
 */
export const DARK_BASEMAP_STYLE =
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

/** Initial map view: tightly framed on the Bay Area. */
export const INITIAL_VIEW = {
  center: [-122.27, 37.62] as [number, number],
  zoom: 9.1,
  bearing: 0,
  pitch: 32,
};

/** Hard bounds so users can't pan into the Pacific by accident. */
export const MAX_BOUNDS: [[number, number], [number, number]] = [
  [-123.2, 36.8],
  [-121.3, 38.4],
];
