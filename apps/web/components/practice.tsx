"use client";

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import { api, call, type QaLevel, type Question } from "@/lib/api";

import { CheckIcon, PlayIcon, RefreshIcon, SparkIcon, XIcon } from "./icons";
import { Markdown } from "./notes-view";

const LEVELS: { value: QaLevel; label: string; hint: string }[] = [
  { value: "mixed", label: "Mixed", hint: "Recall and application" },
  { value: "recall", label: "Recall", hint: "What the notes say" },
  { value: "application", label: "Apply", hint: "Use it on a new case" },
];

type Verdict = "known" | "review";

/** Practice questions grounded in the notes. Answers stay hidden until asked for; the
 * student marks each "Got it" or "Review" and sees a running score. Same settings again
 * are served from the cache (instant, free). */
export function Practice({ notesId }: { notesId: string }) {
  const [level, setLevel] = useState<QaLevel>("mixed");
  const [count, setCount] = useState(5);
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});

  const generate = useMutation({
    mutationFn: () =>
      call(
        api.POST("/v1/notes/{notes_id}/qa", {
          params: { path: { notes_id: notesId } },
          body: { level, count },
        }),
      ),
    onSuccess: () => setVerdicts({}),
  });

  const questions = generate.data?.questions ?? [];
  const known = Object.values(verdicts).filter((v) => v === "known").length;

  return (
    <div>
      <div className="practice-bar">
        <div className="group">
          Level
          <div className="chips">
            {LEVELS.map((l) => (
              <button
                key={l.value}
                type="button"
                className="chip"
                aria-pressed={level === l.value}
                title={l.hint}
                onClick={() => setLevel(l.value)}
              >
                {l.label}
              </button>
            ))}
          </div>
        </div>
        <div className="group">
          Questions
          <div className="chips">
            {[3, 5, 10].map((n) => (
              <button
                key={n}
                type="button"
                className="chip"
                aria-pressed={count === n}
                onClick={() => setCount(n)}
              >
                {n}
              </button>
            ))}
          </div>
        </div>
        <button
          type="button"
          className="btn"
          disabled={generate.isPending}
          onClick={() => generate.mutate()}
        >
          {generate.isPending ? (
            <>
              <RefreshIcon className="spin" /> Writing questions…
            </>
          ) : (
            <>
              <SparkIcon /> {questions.length ? "New set" : "Generate questions"}
            </>
          )}
        </button>
        {questions.length > 0 && (
          <span className="score">
            {known}/{questions.length} known
          </span>
        )}
      </div>

      {generate.isError && <p className="alert error">{generate.error.message}</p>}

      {!generate.data && !generate.isPending && (
        <div className="card empty">
          <SparkIcon size={28} />
          <h3>Test yourself on this lesson</h3>
          <p>Questions come only from your notes, and every answer links to where it’s taught.</p>
        </div>
      )}
      {generate.data && questions.length === 0 && (
        <p className="alert warn">No questions could be grounded in these notes.</p>
      )}

      {questions.map((q, i) => (
        <QuestionCard
          key={q.id}
          number={i + 1}
          question={q}
          verdict={verdicts[q.id]}
          onVerdict={(v) => setVerdicts({ ...verdicts, [q.id]: v })}
        />
      ))}
    </div>
  );
}

function QuestionCard({
  number,
  question,
  verdict,
  onVerdict,
}: {
  number: number;
  question: Question;
  verdict: Verdict | undefined;
  onVerdict: (v: Verdict) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <article className={`qcard ${verdict ?? ""}`}>
      <div className="qhead">
        <span className="qnum">{number}</span>
        <span className={`badge ${question.level === "application" ? "warn" : "accent"}`}>
          {question.level === "application" ? "Apply" : "Recall"}
        </span>
        {verdict === "known" && <span className="badge ok">Got it</span>}
        {verdict === "review" && <span className="badge warn">Review</span>}
      </div>
      <div className="qtext">
        <Markdown>{question.question}</Markdown>
      </div>

      {open ? (
        <div className="answer">
          <Markdown>{question.answer}</Markdown>
          <div className="answer-foot">
            From
            {question.units.map((u) => (
              <span key={u.id} className="badge">
                {u.concept}
              </span>
            ))}
            {question.timestamps.map((t) => (
              <a
                key={t.url}
                className="time primary"
                href={t.url}
                target="_blank"
                rel="noopener noreferrer"
              >
                <PlayIcon size={10} /> {t.label}
              </a>
            ))}
          </div>
          <div className="self-check">
            <button type="button" className="btn ghost sm" onClick={() => onVerdict("known")}>
              <CheckIcon size={14} /> Got it
            </button>
            <button type="button" className="btn ghost sm" onClick={() => onVerdict("review")}>
              <XIcon size={14} /> Review this
            </button>
          </div>
        </div>
      ) : (
        <div className="self-check">
          <button type="button" className="btn ghost sm" onClick={() => setOpen(true)}>
            Show answer
          </button>
        </div>
      )}
    </article>
  );
}
