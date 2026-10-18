# Security policy

## Reporting a vulnerability

Please **don't open a public issue**. Report it privately through
[GitHub security advisories](https://github.com/Dev-yadav2003/StudyLoop/security/advisories/new)
with steps to reproduce and the impact you expect. You'll get a reply within a few days.

## How the project handles sensitive data

- **Secrets** live only in `.env` / `apps/web/.env.local` (git-ignored) and are typed as
  `SecretStr`, so they never appear in logs. The API holds no Supabase secret: sign-in
  tokens are verified with the project's public keys.
- **Access control:** every library query is scoped to the signed-in account; another
  account's IDs return 404. Editing shared notes is limited to owner accounts.
- **Usage limits:** guest accounts have a lifetime allowance, checked before any work starts.
- **Untrusted model output** is schema-validated; quoted code and timestamps are checked
  against the lecture, not trusted.
- **Untrusted code** from lectures is never executed by the app. The eval suite runs it only
  on request, in CI inside a locked-down container (no network, read-only filesystem,
  resource limits, unprivileged user).
