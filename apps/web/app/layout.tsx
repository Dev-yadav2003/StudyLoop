import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "Lecture Notes",
  description: "Turn a YouTube lecture into faithful, timestamped study notes.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="site-header">
          <Link href="/" className="brand">
            Lecture Notes
          </Link>
        </header>
        <main className="container">
          <Providers>{children}</Providers>
        </main>
      </body>
    </html>
  );
}
