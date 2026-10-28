// Typed client generated from FastAPI's OpenAPI schema (`make client`).
// If the API changes shape, src/schema.d.ts changes and `tsc` fails here first.
import createClient from "openapi-fetch";
import type { components, paths } from "./schema";

export const api = createClient<paths>({ baseUrl: "" });

type Schemas = components["schemas"];
export type PieceSummary = Schemas["PieceSummary"];
export type PieceDetail = Schemas["PieceDetail"];
export type Segment = Schemas["SegmentOut"];
export type Hit = Schemas["HitOut"];
export type Chunk = Schemas["ChunkOut"];
export type Audit = Schemas["AuditOut"];

/** The server's explanation for a refused or invalid request. */
export function errorText(error: unknown): string {
  const detail = (error as { detail?: unknown })?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg ?? JSON.stringify(d)).join("; ");
  return "Request failed";
}
