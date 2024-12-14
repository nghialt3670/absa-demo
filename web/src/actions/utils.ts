import puppeteer from "puppeteer";

export async function getHtml(url: string): Promise<string> {
  const browser = await puppeteer.launch({
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu"],
  });

  try {
    const page = await browser.newPage();

    await page.setUserAgent(
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    );

    await page.goto(url, {
      waitUntil: "networkidle0",
      timeout: 15000,
    });

    return await page.content();
  } finally {
    await browser.close();
  }
}
