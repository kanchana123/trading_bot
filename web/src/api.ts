const KEY = "tb-api-key";
const OPERATOR = "tb-operator";

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
  if (typeof data === "string") return data;
  if (data && typeof data === "object" && "detail" in data) {
    const value = (data as { detail: unknown }).detail;
    if (typeof value === "string") return value;
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
  const res = await fetch(path, { ...init, headers });
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) throw new Error(detail(data, res.statusText));
  return data as T;
}
