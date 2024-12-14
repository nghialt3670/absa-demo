import { createContext } from "react";

export const MapContext = createContext<mapboxgl.Map | null>(null);
