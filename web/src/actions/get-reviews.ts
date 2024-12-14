"use server";

import { Place } from "@/interfaces/place";
import { v4 } from "uuid";
import { load } from "cheerio";
import { env } from "@/lib/env";
import to from "await-to-js";
import { Review } from "@/interfaces/review";
import { getHtml } from "./utils";

export async function getReviewsByScrappingHtml(
  href: string,
): Promise<Review[]> {
  const html = await getHtml(href);
  const $ = load(html);
  const spans = $("span.wiI7pd");
  const reviews: Review[] = [];

  spans.each((_, element) => {
    const id = v4();
    const text = $(element).text();
    reviews.push({ id, text });
  });

  return reviews;
}

export async function getReviewsByScrappingDog(
  dataId: string,
  apiKey: string,
): Promise<Review[]> {
  const endpoint = new URL("https://api.scrapingdog.com/google_maps/reviews");

  endpoint.searchParams.append("api_key", apiKey);
  endpoint.searchParams.append("data_id", dataId);

  const response = await fetch(endpoint);
  const payload = await response.json();
  const results: any[] = payload["reviews_results"];

  return results.map((result) => ({
    id: v4(),
    text: result["snippet"],
  }));
}

export async function getReviews(place: Place): Promise<Review[]> {
  if (place.dataId) {
    const apiKeys = env.SCRAPPING_DOG_API_KEYS.split(",");

    for (const apiKey of apiKeys) {
      const [error, reviews] = await to(
        getReviewsByScrappingDog(place.dataId, apiKey),
      );

      if (error) {
        continue;
      }

      return reviews;
    }
  }

  if (place.href) {
    return getReviewsByScrappingHtml(place.href);
  }

  throw new Error("Invalid place");
}
