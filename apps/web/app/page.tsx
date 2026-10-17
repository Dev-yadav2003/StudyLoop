"use client";

import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

import { api, toApiError, type SubjectPreset } from "@/lib/api";

const PRESETS: { value: SubjectPreset; label: string }[] = [
  { value: "general", label: "General" },
  { value: "stem", label: "STEM — formulas and code first" },
  { value: "theory", label: "Theory — arguments and definitions first" },
];

export default function SubmitPage() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [preset, setPreset] = useState<SubjectPreset>("general");

  const submit = useMutation({
    mutationFn: async () => {
      const { data, error, response } = await api.POST("/v1/jobs", {
        body: { url: url.trim(), subject_preset: preset },
      });
      if (error || !data) throw toApiError(error, response.status);
      return data;
    },
    onSuccess: (job) => router.push(`/jobs/${job.id}`),
  });

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    submit.mutate();
  };

  return (
    <>
      <h1>Notes from a YouTube lecture</h1>
      <p className="muted">
        Paste a link to one public video. You get notes of what the teacher said and what was on
        the board, with timestamp links.
      </p>
      <form className="card" onSubmit={onSubmit}>
        <label>
          Video link
          <input
            type="url"
            required
            placeholder="https://www.youtube.com/watch?v=…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        </label>
        <label>
          Subject
          <select value={preset} onChange={(e) => setPreset(e.target.value as SubjectPreset)}>
            {PRESETS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        {submit.isError && (
          <p className="alert error" role="alert">
            {submit.error.message}
          </p>
        )}
        <div>
          <button type="submit" disabled={submit.isPending}>
            {submit.isPending ? "Submitting…" : "Make notes"}
          </button>
        </div>
      </form>
    </>
  );
}
