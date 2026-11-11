const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api").replace(
  /\/$/,
  "",
);

export async function request(path, options = {}) {
  const controller = new AbortController();
  const cancel = () => controller.abort(options.signal.reason);
  if (options.signal?.aborted) cancel();
  options.signal?.addEventListener("abort", cancel, { once: true });
  const timer = setTimeout(
    () =>
      controller.abort(
        new DOMException(
          "Analysis timed out. Please try again.",
          "TimeoutError",
        ),
      ),
    path.endsWith("/refresh") ? 180000 : 45000,
  );
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
    });
    const body = await response.json();
    if (!response.ok)
      throw new Error(body.error || `Request failed (${response.status})`);
    return body;
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", cancel);
  }
}

export function downloadJson(value, name) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], {
      type: "application/geo+json",
    }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function number(value, digits = 0) {
  return value == null || !Number.isFinite(Number(value))
    ? "\u2014"
    : Number(value).toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });
}

export const scenarioLabel = (name) =>
  ({ plus_20cm: "+20 cm", plus_50cm: "+50 cm", plus_100cm: "+100 cm" })[name] ||
  name;

export function dateLabel(value) {
  return value
    ? new Intl.DateTimeFormat("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        timeZone: "UTC",
      }).format(new Date(value))
    : "Not recorded";
}
