"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";

import { KeepAlive } from "@/components/keep-alive";

export function Providers({ children }: { children: ReactNode }) {
  // One client per browser session (useState keeps it stable across re-renders).
  const [client] = useState(
    () => new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false } } }),
  );
  return (
    <QueryClientProvider client={client}>
      <KeepAlive />
      {children}
    </QueryClientProvider>
  );
}
