"use server";

import { Place } from "@/interfaces/place";
import { v4 } from "uuid";
import { load } from "cheerio";
import { env } from "@/lib/env";
import to from "await-to-js";
import { getHtml } from "./utils";

export async function getPlacesByScrappingHtml(
  query: string,
  longitude: number,
  latitude: number,
): Promise<Place[]> {
  const endpoint = new URL(
    `https://www.google.com/maps/search/${query}/@${latitude.toFixed(7)},${longitude.toFixed(7)},12z`,
  );
  const html = await getHtml(endpoint.toString());
  const $ = load(html);
  const links = $("a.hfpxzc[href]");
  const places: Place[] = [];

  links.each((_, element) => {
    const name = $(element).attr("aria-label");
    const href = $(element).attr("href");

    if (!name || !href) {
      return;
    }

    const pattern = /!3d([-\d.]+)!4d([-\d.]+)/;
    const match = href.match(pattern);

    if (match?.length !== 3) {
      return;
    }

    const id = v4();
    const latitude = parseFloat(match[1]);
    const longitude = parseFloat(match[2]);

    places.push({ id, name, href, longitude, latitude });
  });

  return places;
}

export async function getPlacesByScrappingDog(
  query: string,
  longitude: number,
  latitude: number,
  apiKey: string,
): Promise<Place[]> {
  const endpoint = new URL("https://api.scrapingdog.com/google_maps");

  endpoint.searchParams.append("api_key", apiKey);
  endpoint.searchParams.append("query", query);
  endpoint.searchParams.append(
    "ll",
    `@${latitude.toFixed(7)},${longitude.toFixed(7)},10z`,
  );
  endpoint.searchParams.append("page", "0");

  const response = await fetch(endpoint);
  const payload = await response.json();
  const results: any[] = payload["search_results"];

  return results.map((result) => ({
    id: v4(),
    name: result["title"],
    longitude: result["gps_coordinates"]["longitude"],
    latitude: result["gps_coordinates"]["latitude"],
    placeId: result["place_id"],
    dataId: result["data_id"],
  }));
}

export async function getPlaces(
  query: string,
  longitude: number,
  latitude: number,
): Promise<Place[]> {
  const apiKeys = env.SCRAPPING_DOG_API_KEYS.split(",");

  for (const apiKey of apiKeys) {
    const [error, places] = await to(
      getPlacesByScrappingDog(query, longitude, latitude, apiKey),
    );

    if (error) {
      continue;
    }

    return places;
  }

  return getPlacesByScrappingHtml(query, longitude, latitude);
}
