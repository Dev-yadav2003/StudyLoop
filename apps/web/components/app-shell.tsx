"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { BookIcon, MenuIcon } from "./icons";
import { Sidebar } from "./sidebar";

/** Sidebar + main area. On narrow screens the sidebar slides in from a menu button. */
export function AppShell({ children }: { children: ReactNode }) {
  const [navOpen, setNavOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => setNavOpen(false), [pathname]);

  return (
    <div className={`shell ${navOpen ? "nav-open" : ""}`}>
      <Sidebar onNavigate={() => setNavOpen(false)} />
      <div className="scrim" onClick={() => setNavOpen(false)} />
      <div className="main">
        <header className="topbar">
          <button
            type="button"
            className="icon-btn"
            aria-label="Open library"
            onClick={() => setNavOpen(true)}
          >
            <MenuIcon size={18} />
          </button>
          <Link href="/" className="brand">
            <span className="brand-mark">
              <BookIcon size={15} />
            </span>
            LecNotes
          </Link>
        </header>
        <main className="page">{children}</main>
      </div>
    </div>
  );
}
