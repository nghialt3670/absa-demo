"use client";

import { useEffect, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { POPUP_HTML, US_BOUNDS } from "./constants";
import { Place } from "@/interfaces/place";
import { MapContext } from "@/contexts/map-context";
import { PlaceMarker } from "@/components/place-marker";
import { useToast } from "@/hooks/use-toast";
import to from "await-to-js";
import { getPlaces } from "@/actions/get-places";

mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN || "";

export default function USMapPage() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<mapboxgl.Map>(null);
  const [places, setPlaces] = useState<Place[]>([]);
  const [loading, setLoading] = useState(false); // Add loading state
  const { toast } = useToast();

  useEffect(() => {
    if (mapContainerRef.current && !mapRef.current) {
      mapRef.current = new mapboxgl.Map({
        container: mapContainerRef.current,
        style: "mapbox://styles/mapbox/streets-v11",
        center: [-98.5795, 39.8283], // Center of the United States
        zoom: 4,
        maxBounds: US_BOUNDS,
        minZoom: 3,
        maxZoom: 18,
      });

      mapRef.current.addControl(
        new mapboxgl.NavigationControl({
          showCompass: true,
        }),
      );

      mapRef.current.on("click", (e) => {
        if (mapRef.current?.getCanvas().style.cursor === "pointer") {
          return;
        }

        const popup = new mapboxgl.Popup({ closeOnClick: true })
          .setLngLat(e.lngLat)
          .setHTML(POPUP_HTML)
          .addTo(mapRef.current!);

        const popupEl = popup.getElement();

        if (!popupEl) {
          return;
        }

        popupEl
          .querySelector("#search-button")
          ?.addEventListener("click", async () => {
            const query = (
              popupEl.querySelector("#search-input") as HTMLInputElement
            )?.value;
            const latitude = popup.getLngLat().lat;
            const longitude = popup.getLngLat().lng;

            popup.remove(); // Close the popup immediately
            setLoading(true); // Set loading state to true

            const [error, places] = await to(
              getPlaces(query, longitude, latitude),
            );

            setLoading(false); // Reset loading state to false

            if (error) {
              toast({
                title: "Searching failed!",
                description: `Failed to search nearby places with the query: "${query}".`,
              });

              return;
            }

            const usPlaces = places.filter((place) =>
              US_BOUNDS.contains([place.longitude, place.latitude]),
            ).slice(0, 10);

            if (usPlaces.length === 0) {
              toast({
                title: "Searching done!",
                description: `No nearby places found with the query: "${query}".`,
              });
            } else {
              toast({
                title: "Searching done!",
                description: `Found ${usPlaces.length} nearby places with the keyword: "${query}".`,
              });

              mapRef.current?.flyTo({
                center: [usPlaces[0].longitude, usPlaces[0].latitude],
                zoom: 10,
              });
            }

            setPlaces(usPlaces);
          });
      });
    }

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  return (
    <MapContext.Provider value={mapRef.current}>
      <div className="relative w-full h-screen">
        {loading && (
          <div className="absolute inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50">
            <p className="text-white text-lg font-bold">Searching...</p>
          </div>
        )}
        <div ref={mapContainerRef} className="size-full" />
        {places.map((place) => (
          <PlaceMarker key={place.id} place={place} />
        ))}
      </div>
    </MapContext.Provider>
  );
}
