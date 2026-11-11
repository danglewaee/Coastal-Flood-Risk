import React, { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import {
  CircleMarker,
  GeoJSON,
  MapContainer,
  ScaleControl,
  TileLayer,
  Tooltip,
  ZoomControl,
  useMap,
} from "react-leaflet";

function Viewport({ analysis, city, selectedId, fitRequest }) {
  const map = useMap();
  const firstView = useRef(true);
  useEffect(() => {
    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(map.getContainer());
    return () => observer.disconnect();
  }, [map]);
  useEffect(() => {
    const selected = analysis?.flood.features.find(
      (f) => f.properties.component_id === selectedId,
    );
    if (selected) {
      map.flyToBounds(L.geoJSON(selected).getBounds(), {
        padding: [70, 70],
        maxZoom: 16,
        duration: 0.65,
      });
    } else if (analysis?.bounds) {
      const [west, south, east, north] = analysis.bounds;
      map.fitBounds(
        [
          [south, west],
          [north, east],
        ],
        { padding: [45, 45], maxZoom: 13, animate: !firstView.current },
      );
    } else if (city?.position.every((value) => value != null)) {
      map.setView(city.position, 11);
    }
    firstView.current = false;
  }, [
    map,
    analysis?.version,
    analysis?.city,
    analysis?.scenario,
    selectedId,
    fitRequest,
  ]);
  return null;
}

export default function FloodMap({
  analysis,
  city,
  scenario,
  layers,
  baseMap,
  selectedId,
  fitRequest,
  onSelect,
}) {
  const [tileError, setTileError] = useState(false);
  const pointStyle = {
    radius: 4,
    color: "#fff",
    weight: 1,
    fillColor: "#ce5245",
    fillOpacity: 1,
  };
  const representativePoints = useMemo(
    () =>
      analysis?.hotspots.map((feature) => {
        return {
          id: feature.properties.component_id,
          rank: feature.properties.rank,
          position: feature.properties.position,
        };
      }) || [],
    [analysis],
  );
  useEffect(() => setTileError(false), [baseMap]);

  return (
    <>
      <MapContainer
        center={[42.36, -71.06]}
        zoom={11}
        zoomControl={false}
        scrollWheelZoom
        className="flood-map"
        aria-label="Coastal flood map"
      >
        <TileLayer
          key={baseMap}
          url={
            baseMap === "satellite"
              ? "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              : "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          }
          attribution={
            baseMap === "satellite"
              ? "Tiles &copy; Esri"
              : '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          }
          eventHandlers={{
            tileerror: () => setTileError(true),
            tileload: () => setTileError(false),
          }}
        />
        <Viewport
          analysis={analysis}
          city={city}
          selectedId={selectedId}
          fitRequest={fitRequest}
        />
        <ZoomControl position="bottomright" />
        <ScaleControl position="bottomleft" imperial={false} />
        {analysis && layers.flood && (
          <GeoJSON
            key={`${analysis.version}:${scenario}:${selectedId}`}
            data={analysis.flood}
            style={(feature) => ({
              fillColor:
                feature.properties.component_id === selectedId
                  ? "#e8b84b"
                  : "#087f83",
              fillOpacity:
                feature.properties.component_id === selectedId ? 0.65 : 0.55,
              color:
                feature.properties.component_id === selectedId
                  ? "#815a0e"
                  : "#086567",
              weight:
                feature.properties.component_id === selectedId ? 2.5 : 0.9,
            })}
            onEachFeature={(feature, layer) => {
              layer.bindTooltip(
                `Region ${feature.properties.component_id} | ${(feature.properties.area_m2 / 10000).toFixed(2)} ha`,
                { sticky: true },
              );
              layer.on("click", () =>
                onSelect(feature.properties.component_id),
              );
            }}
          />
        )}
        {analysis && layers.infrastructure && (
          <GeoJSON
            key={`exposure:${analysis.version}`}
            data={analysis.exposure.features}
            style={{ color: "#cc5142", weight: 3 }}
            pointToLayer={(_, latlng) => L.circleMarker(latlng, pointStyle)}
            onEachFeature={(feature, layer) => {
              const label = document.createElement("span");
              label.textContent = feature.properties.name;
              layer.bindTooltip(label);
            }}
          />
        )}
        {layers.hotspots &&
          representativePoints.map((point) => (
            <CircleMarker
              key={point.id}
              center={point.position}
              radius={point.id === selectedId ? 8 : 5}
              pathOptions={{
                color: "#fff",
                weight: 2,
                fillColor: "#ce5245",
                fillOpacity: 1,
              }}
              eventHandlers={{ click: () => onSelect(point.id) }}
            >
              <Tooltip>
                Region {point.id} · Area rank #{point.rank}
              </Tooltip>
            </CircleMarker>
          ))}
      </MapContainer>
      {tileError && (
        <div className="tile-status" role="status">
          Basemap unavailable. Flood layers remain interactive.
        </div>
      )}
    </>
  );
}
