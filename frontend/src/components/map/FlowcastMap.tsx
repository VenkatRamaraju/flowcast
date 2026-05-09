import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl, { type MapLayerMouseEvent } from "maplibre-gl";
import { AnimatePresence, motion } from "framer-motion";

import type { Station } from "@/types";
import { useAppStore } from "@/store/useAppStore";
import { DARK_BASEMAP_STYLE, INITIAL_VIEW, MAX_BOUNDS } from "./mapStyle";

interface Props {
  stations: Station[];
}

const SOURCE_ID = "stations-src";
const DOT_LAYER = "stations-dot";
const GLOW_LAYER = "stations-glow";
const HALO_LAYER = "stations-halo";

function toFeatureCollection(stations: Station[]): GeoJSON.FeatureCollection {
  return {
    type: "FeatureCollection",
    features: stations.map((s) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [s.lon, s.lat] },
      properties: { id: s.id, region: s.region },
    })),
  };
}

export function FlowcastMap({ stations }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const [styleLoaded, setStyleLoaded] = useState(false);

  const selectedId = useAppStore((s) => s.selectedStationId);
  const hoveredId = useAppStore((s) => s.hoveredStationId);
  const selectStation = useAppStore((s) => s.selectStation);
  const setHovered = useAppStore((s) => s.setHovered);

  const stationsById = useMemo(() => {
    const m = new Map<string, Station>();
    for (const s of stations) m.set(s.id, s);
    return m;
  }, [stations]);

  // Boot the map exactly once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: DARK_BASEMAP_STYLE,
      center: INITIAL_VIEW.center,
      zoom: INITIAL_VIEW.zoom,
      bearing: INITIAL_VIEW.bearing,
      pitch: INITIAL_VIEW.pitch,
      maxBounds: MAX_BOUNDS,
      minZoom: 8.2,
      maxZoom: 17,
      attributionControl: false,
      antialias: true,
      fadeDuration: 250,
    });

    map.dragRotate.disable();
    map.touchZoomRotate.disableRotation();
    map.addControl(
      new maplibregl.NavigationControl({ visualizePitch: false, showCompass: false }),
      "top-right",
    );

    map.on("load", () => setStyleLoaded(true));
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Push station data into the map once the style is ready (or when stations change).
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !styleLoaded) return;

    const data = toFeatureCollection(stations);
    const existing = map.getSource(SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
    if (existing) {
      existing.setData(data);
      return;
    }

    map.addSource(SOURCE_ID, { type: "geojson", data, promoteId: "id" });

    // Soft outer glow that grows with zoom.
    map.addLayer({
      id: GLOW_LAYER,
      type: "circle",
      source: SOURCE_ID,
      paint: {
        "circle-radius": [
          "interpolate",
          ["linear"],
          ["zoom"],
          9, 4.5,
          12, 9,
          15, 18,
        ],
        "circle-color": "#34e3c2",
        "circle-opacity": 0.10,
        "circle-blur": 1.0,
      },
    });

    // Hover halo — only painted for the hovered feature via a feature-state flag.
    map.addLayer({
      id: HALO_LAYER,
      type: "circle",
      source: SOURCE_ID,
      paint: {
        "circle-radius": [
          "interpolate",
          ["linear"],
          ["zoom"],
          9, 6.5,
          12, 11,
          15, 22,
        ],
        "circle-color": "#7af0d8",
        "circle-opacity": [
          "case",
          ["boolean", ["feature-state", "hover"], false],
          0.35,
          0,
        ],
        "circle-blur": 0.6,
      },
    });

    // Crisp dot.
    map.addLayer({
      id: DOT_LAYER,
      type: "circle",
      source: SOURCE_ID,
      paint: {
        "circle-radius": [
          "interpolate",
          ["linear"],
          ["zoom"],
          9, 2.4,
          12, 4.0,
          15, 7.5,
        ],
        "circle-color": "#e9fff8",
        "circle-stroke-color": "#34e3c2",
        "circle-stroke-width": [
          "interpolate",
          ["linear"],
          ["zoom"],
          9, 1.0,
          15, 1.6,
        ],
        "circle-opacity": [
          "case",
          ["boolean", ["feature-state", "hover"], false],
          1.0,
          0.92,
        ],
      },
    });

    let activeHoverId: string | null = null;

    const onMouseMove = (e: MapLayerMouseEvent) => {
      const feat = e.features?.[0];
      if (!feat) return;
      map.getCanvas().style.cursor = "pointer";
      const id = feat.properties?.id as string | undefined;
      if (!id || id === activeHoverId) return;
      if (activeHoverId) {
        map.setFeatureState({ source: SOURCE_ID, id: activeHoverId }, { hover: false });
      }
      activeHoverId = id;
      map.setFeatureState({ source: SOURCE_ID, id }, { hover: true });
      setHovered(id);
    };

    const onMouseLeave = () => {
      map.getCanvas().style.cursor = "";
      if (activeHoverId) {
        map.setFeatureState({ source: SOURCE_ID, id: activeHoverId }, { hover: false });
        activeHoverId = null;
      }
      setHovered(null);
    };

    const onClick = (e: MapLayerMouseEvent) => {
      const feat = e.features?.[0];
      const id = feat?.properties?.id as string | undefined;
      if (!id) return;
      selectStation(id);
    };

    map.on("mousemove", DOT_LAYER, onMouseMove);
    map.on("mousemove", HALO_LAYER, onMouseMove);
    map.on("mouseleave", DOT_LAYER, onMouseLeave);
    map.on("mouseleave", HALO_LAYER, onMouseLeave);
    map.on("click", DOT_LAYER, onClick);
    map.on("click", HALO_LAYER, onClick);

    return () => {
      map.off("mousemove", DOT_LAYER, onMouseMove);
      map.off("mousemove", HALO_LAYER, onMouseMove);
      map.off("mouseleave", DOT_LAYER, onMouseLeave);
      map.off("mouseleave", HALO_LAYER, onMouseLeave);
      map.off("click", DOT_LAYER, onClick);
      map.off("click", HALO_LAYER, onClick);
    };
  }, [styleLoaded, stations, selectStation, setHovered]);

  // Fly to the selected station whenever it changes.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !selectedId) return;
    const station = stationsById.get(selectedId);
    if (!station) return;
    map.easeTo({
      center: [station.lon, station.lat],
      zoom: Math.max(map.getZoom(), 13.4),
      duration: 900,
      easing: (t) => 1 - Math.pow(1 - t, 3),
      padding: { right: 440, top: 0, bottom: 0, left: 0 },
    });
  }, [selectedId, stationsById]);

  // Dismiss selection when clicking on the empty map (not on a station).
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !styleLoaded) return;
    const handler = (e: MapLayerMouseEvent) => {
      const hits = map.queryRenderedFeatures(e.point, {
        layers: [DOT_LAYER, HALO_LAYER],
      });
      if (hits.length === 0) selectStation(null);
    };
    map.on("click", handler);
    return () => {
      map.off("click", handler);
    };
  }, [styleLoaded, selectStation]);

  const selected = selectedId ? stationsById.get(selectedId) : null;
  const hovered = hoveredId && hoveredId !== selectedId ? stationsById.get(hoveredId) : null;

  return (
    <div className="absolute inset-0">
      <div ref={containerRef} className="absolute inset-0" />
      {/* Vignette + top fade for premium framing */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_55%,rgba(0,0,0,0.55)_100%)]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-ink-950/80 to-transparent" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-ink-950/70 to-transparent" />

      {/* Animated overlay for the selected station — DOM marker on top of canvas */}
      <MapMarkerOverlay map={mapRef} station={selected} variant="selected" />
      <MapMarkerOverlay map={mapRef} station={hovered} variant="hover" />

      {/* Subtle credit */}
      <div className="pointer-events-none absolute bottom-3 right-4 text-[10px] uppercase tracking-[0.2em] text-white/30">
        © OpenStreetMap · CARTO
      </div>
    </div>
  );
}

interface OverlayProps {
  map: React.MutableRefObject<maplibregl.Map | null>;
  station: Station | null | undefined;
  variant: "selected" | "hover";
}

function MapMarkerOverlay({ map, station, variant }: OverlayProps) {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const m = map.current;
    if (!m || !station) {
      setPos(null);
      return;
    }
    const project = () => {
      const p = m.project([station.lon, station.lat]);
      setPos({ x: p.x, y: p.y });
    };
    project();
    m.on("move", project);
    m.on("zoom", project);
    m.on("rotate", project);
    return () => {
      m.off("move", project);
      m.off("zoom", project);
      m.off("rotate", project);
    };
  }, [map, station]);

  return (
    <AnimatePresence>
      {station && pos && (
        <motion.div
          key={`${variant}-${station.id}`}
          className="pointer-events-none absolute z-10"
          style={{ left: pos.x, top: pos.y }}
          initial={{ opacity: 0, scale: 0.7 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.7 }}
          transition={{ type: "spring", stiffness: 320, damping: 26 }}
        >
          <div className="relative -translate-x-1/2 -translate-y-1/2">
            {variant === "selected" ? (
              <>
                <span className="absolute inset-0 -m-3 rounded-full bg-flow-inflow/20 blur-md" />
                <span className="absolute inset-0 -m-2 rounded-full border border-flow-inflow/40 animate-pulseRing" />
                <span className="absolute inset-0 -m-2 rounded-full border border-flow-inflow/40 animate-pulseRing [animation-delay:1.1s]" />
                <span className="relative block h-3.5 w-3.5 rounded-full bg-flow-inflow shadow-marker" />
              </>
            ) : (
              <span className="block h-2.5 w-2.5 rounded-full bg-white/90 shadow-marker" />
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
