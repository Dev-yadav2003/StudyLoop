"use client";

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { api, type QaLevel, toApiError } from "@/lib/api";

const LEVELS: { value: QaLevel; label: string }[] = [
  { value: "mixed", label: "Mixed" },
  { value: "recall", label: "Recall" },
  { value: "application", label: "Application" },
];

/** Practice questions from the notes. Each answer is hidden until asked for and links to
 * where it is taught in the video. Asking again with the same settings is instant (cached). */
export function PracticeQuestions({ notesId }: { notesId: string }) {
  const [level, setLevel] = useState<QaLevel>("mixed");
  const [count, setCount] = useState(5);

  const generate = useMutation({
    mutationFn: async () => {
      const { data, error, response } = await api.POST("/v1/notes/{notes_id}/qa", {
        params: { path: { notes_id: notesId } },
        body: { level, count },
      });
      if (error || !data) throw toApiError(error, response.status);
      return data;
    },
  });

  return (
    <section className="card practice">
      <h2>Practice questions</h2>
      <form
        className="actions"
        onSubmit={(e) => {
          e.preventDefault();
          generate.mutate();
        }}
      >
        <label>
          Level{" "}
          <select value={level} onChange={(e) => setLevel(e.target.value as QaLevel)}>
            {LEVELS.map((l) => (
              <option key={l.value} value={l.value}>
                {l.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Questions{" "}
          <select value={count} onChange={(e) => setCount(Number(e.target.value))}>
            {[3, 5, 10].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <button type="submit" disabled={generate.isPending}>
          {generate.isPending ? "Writing questions…" : "Generate"}
        </button>
      </form>

      {generate.isError && <p className="alert error">{generate.error.message}</p>}
      {generate.data && generate.data.questions.length === 0 && (
        <p className="alert warn">No questions could be grounded in these notes.</p>
      )}
      <ol className="questions">
        {generate.data?.questions.map((q) => (
          <li key={q.id}>
            <span className={`level level-${q.level}`}>{q.level}</span>
            <div className="notes">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{q.question}</ReactMarkdown>
            </div>
            <details>
              <summary>Show answer</summary>
              <div className="notes">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{q.answer}</ReactMarkdown>
              </div>
              <p className="muted">
                From: {q.units.map((u) => u.concept).join(", ")} ·{" "}
                {q.timestamps.map((t, i) => (
                  <span key={t.url}>
                    {i > 0 && ", "}
                    <a href={t.url} target="_blank" rel="noopener noreferrer">
                      {t.label}
                    </a>
                  </span>
                ))}
              </p>
            </details>
          </li>
        ))}
      </ol>
    </section>
  );
}
