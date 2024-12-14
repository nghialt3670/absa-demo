import { NextRequest } from "next/server";
import puppeteer from "puppeteer";

export async function getFullHtml(url: string): Promise<string> {
  const browser = await puppeteer.launch();

  try {
    const page = await browser.newPage();

    // Optional: Set user agent to mimic a real browser
    await page.setUserAgent(
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    );

    // Navigate to the URL
    await page.goto(url, {
      waitUntil: "networkidle0", // Wait until network is idle
      timeout: 30000, // 30 seconds timeout
    });

    // Get the full page source
    const pageSource = await page.content();

    return pageSource;
  } catch (error) {
    console.error("Error fetching HTML:", error);
    throw error;
  } finally {
    await browser.close();
  }
}

export async function GET(req: NextRequest) {
  const params = new URLSearchParams(req.url);
  const keyword = params.get("keywords");
  const latitude = parseFloat(params.get("latitude")!).toFixed(7);
  const longitude = parseFloat(params.get("longitude")!).toFixed(7);
  const endpoint = new URL(
    `https://www.google.com/maps/search/${keyword}/@${latitude},${longitude},12z`,
  );
}
