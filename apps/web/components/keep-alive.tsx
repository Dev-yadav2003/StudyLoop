"use client";

import { useEffect } from "react";

import { API_URL } from "@/lib/api";

// Render's free plan sleeps the API after 15 minutes without traffic, and waking it takes
// ~1 minute. While the app is open, ping it well inside that window. (Pinging on load also
// starts waking a sleeping API before the user's first real request.)
const INTERVAL_MS = 10 * 60 * 1000;

export function KeepAlive() {
  useEffect(() => {
    const ping = () => {
      fetch(`${API_URL}/healthz`, { cache: "no-store" }).catch(() => {});
    };
    ping();
    const id = setInterval(ping, INTERVAL_MS);
    return () => clearInterval(id);
  }, []);
  return null;
}
