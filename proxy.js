import { timingSafeEqual } from "node:crypto";
import { next } from "@vercel/functions";

// This runs before the static cache. Server code and model archives stay private.
export const config = { runtime: "nodejs" };

export default function proxy(request) {
  const secret = process.env.SMC_INTERNAL_TOKEN ?? "";
  const actual = Buffer.from(request.headers.get("x-smc-internal-token") ?? "");
  const expected = Buffer.from(secret);
  if (secret.length < 32 || actual.length !== expected.length || !timingSafeEqual(actual, expected)) {
    return new Response("Private training source", { status: 403, headers: { "Cache-Control": "private, no-store" } });
  }
  return next();
}
