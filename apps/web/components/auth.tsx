"use client";

import type { Session } from "@supabase/supabase-js";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createContext,
  useContext,
  useEffect,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";

import { api, call } from "@/lib/api";
import { supabase } from "@/lib/supabase";

import { BookIcon, CheckIcon } from "./icons";

const SessionContext = createContext<Session | null>(null);

export const useSession = () => useContext(SessionContext);

/** The signed-in account and its allowance (owner: unlimited). */
export function useMe() {
  const session = useSession();
  return useQuery({
    queryKey: ["me", session?.user.id],
    enabled: Boolean(session),
    queryFn: () => call(api.GET("/v1/me")),
  });
}

export async function signOut(clear: () => void) {
  await supabase?.auth.signOut();
  clear();
}

/** Shows the app only to signed-in users; everyone else gets the sign-in screen. */
export function AuthGate({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [session, setSession] = useState<Session | null | undefined>(undefined);

  useEffect(() => {
    if (!supabase) return;
    supabase.auth.getSession().then(({ data }) => setSession(data.session));
    const { data } = supabase.auth.onAuthStateChange((_event, next) => {
      setSession(next);
      if (!next) qc.clear(); // nothing of the previous account stays cached
    });
    return () => data.subscription.unsubscribe();
  }, [qc]);

  if (!supabase) return <NotConfigured />;
  if (session === undefined) return null; // reading the stored session (a few ms)
  if (session === null) return <SignIn />;
  return <SessionContext.Provider value={session}>{children}</SessionContext.Provider>;
}

function SignIn() {
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!supabase) return;
    setBusy(true);
    setError(null);
    const { error: failed } = await supabase.auth.signInWithOtp({
      email: email.trim(),
      options: { emailRedirectTo: window.location.origin },
    });
    setBusy(false);
    if (failed) setError(failed.message);
    else setSent(true);
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <span className="brand-mark auth-mark">
          <BookIcon size={22} />
        </span>
        <h1>StudyLoop</h1>
        <p className="muted">
          Turn YouTube lectures into clean study notes and practice questions.
        </p>
        {sent ? (
          <div className="auth-sent">
            <CheckIcon size={18} />
            <div>
              <strong>Check your inbox</strong>
              <p className="muted">
                We sent a sign-in link to {email}. Open it on this device to continue.
              </p>
              <button type="button" className="btn quiet sm" onClick={() => setSent(false)}>
                Use another email
              </button>
            </div>
          </div>
        ) : (
          <form className="auth-form" onSubmit={submit}>
            <label className="field">
              Email
              <input
                className="input"
                type="email"
                required
                autoFocus
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </label>
            <button type="submit" className="btn block" disabled={busy || !email.trim()}>
              {busy ? "Sending…" : "Email me a sign-in link"}
            </button>
            {error && <p className="alert error">{error}</p>}
            <p className="muted auth-note">
              No password. New here? The link creates your account. Free accounts include 1 hour of
              lecture video.
            </p>
          </form>
        )}
      </div>
    </div>
  );
}

function NotConfigured() {
  return (
    <div className="auth-page">
      <div className="auth-card">
        <h1>Sign-in isn’t configured</h1>
        <p className="muted">
          Set <code>NEXT_PUBLIC_SUPABASE_URL</code> and{" "}
          <code>NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY</code> in <code>apps/web/.env.local</code>,
          then restart <code>npm run dev</code>.
        </p>
      </div>
    </div>
  );
}
