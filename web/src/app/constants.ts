import { LngLatBounds } from "mapbox-gl";

export const US_BOUNDS: LngLatBounds = new LngLatBounds([
  -125.0011, // Southwest longitude
  24.9493, // Southwest latitude
  -66.9326, // Northeast longitude
  49.5904, // Northeast latitude
]);

export const POPUP_HTML = `
  <div style="
    font-family: Arial, sans-serif;
    width: 200px;
    padding: 10px;
    border-radius: 8px;
    background-color: #ffffff;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
  ">
    <h3 style="margin: 0 0 10px; font-size: 16px; color: #333;">Search nearby</h3>
    <input
      type="text"
      id="search-input"
      style="
        width: 100%;
        margin-top: 5px;
        margin-bottom: 10px;
        padding: 8px;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 14px;
      "
    />
    <button
      id="search-button"
      style="
        display: block;
        width: 100%;
        padding: 8px 10px;
        background-color: #007bff;
        color: white;
        font-size: 14px;
        font-weight: bold;
        text-align: center;
        border: none;
        border-radius: 4px;
        cursor: pointer;
      "
    >
      Search
    </button>
  </div>
  `;
