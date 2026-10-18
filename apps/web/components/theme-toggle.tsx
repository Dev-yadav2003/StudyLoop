"use client";

import { useEffect, useState } from "react";

import { MoonIcon, SunIcon } from "./icons";

type Theme = "light" | "dark";
const KEY = "lecnotes-theme";

function systemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Light/dark switch. The choice is remembered per browser; until then the system decides. */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    let saved: string | null = null;
    try {
      saved = localStorage.getItem(KEY);
    } catch {
      // storage blocked (private mode): fall back to the system theme
    }
    setTheme(saved === "light" || saved === "dark" ? saved : systemTheme());
  }, []);

  if (theme === null) return <span style={{ width: 28 }} />;

  const next: Theme = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      className="icon-btn"
      title={`Switch to ${next} mode`}
      aria-label={`Switch to ${next} mode`}
      onClick={() => {
        document.documentElement.dataset.theme = next;
        try {
          localStorage.setItem(KEY, next);
        } catch {
          // not remembered, still applied
        }
        setTheme(next);
      }}
    >
      {theme === "dark" ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}

/** Inline script for <head>: apply the saved theme before first paint (no flash). */
export const THEME_SCRIPT = `try{var t=localStorage.getItem("${KEY}");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;
