const KEY = "tb-api-key";
const OPERATOR = "tb-operator";

export const API_UNAVAILABLE =
  "No trading API on this host. How it works is static; run FastAPI locally for live runs.";

export function isApiUnavailable(err: unknown) {
  return err instanceof Error && err.message === API_UNAVAILABLE;
}

function looksLikeHtml(text: string) {
  const head = text.trimStart().slice(0, 240).toLowerCase();
  return (
    head.startsWith("<!doctype") ||
    head.startsWith("<html") ||
    head.includes("github pages") ||
    head.includes("file not found")
  );
}

export function getApiKey() {
  return localStorage.getItem(KEY) || "";
}

export function setApiKey(value: string) {
  localStorage.setItem(KEY, value);
}

export function getOperator() {
  return localStorage.getItem(OPERATOR) || "desk-operator";
}

export function setOperator(value: string) {
  localStorage.setItem(OPERATOR, value);
}

function detail(data: unknown, fallback: string) {
  if (typeof data === "string") {
    if (looksLikeHtml(data) || data.length > 280) return API_UNAVAILABLE;
    return data;
  }
  if (data && typeof data === "object" && "detail" in data) {
    const value = (data as { detail: unknown }).detail;
    if (typeof value === "string") {
      if (looksLikeHtml(value) || value.length > 280) return API_UNAVAILABLE;
      return value;
    }
  }
  return fallback;
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init.headers as Record<string, string> | undefined),
  };
  const key = getApiKey().trim();
  if (key) headers["X-API-Key"] = key;
  const base = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");
  const res = await fetch(`${base}${path}`, { ...init, headers });
  const text = await res.text();
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("text/html") || looksLikeHtml(text)) {
    throw new Error(API_UNAVAILABLE);
  }
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) throw new Error(detail(data, res.statusText));
  return data as T;
}
