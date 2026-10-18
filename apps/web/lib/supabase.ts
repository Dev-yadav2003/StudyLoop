// Supabase Auth in the browser: email sign-in link, session kept in localStorage and
// refreshed automatically. Only the publishable (anon) key is used here — it is safe to ship
// to browsers; the API verifies every request's token with the project's public keys.
import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const key = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY;

export const supabase: SupabaseClient | null =
  url && key
    ? createClient(url, key, {
        auth: { flowType: "pkce", detectSessionInUrl: true, persistSession: true },
      })
    : null;

export async function accessToken(): Promise<string | null> {
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession();
  return data.session?.access_token ?? null;
}
