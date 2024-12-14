import z from "zod";

const envSchema = z.object({
  NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN: z.string(),
  NEXT_PUBLIC_API_URL: z.string().url(),
  SCRAPPING_DOG_API_KEYS: z.string(),
});

export const env = envSchema.parse(process.env);
