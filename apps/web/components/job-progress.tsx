"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { ApiError, api, call, type Job } from "@/lib/api";

import { CheckIcon, XIcon } from "./icons";

const POLL_MS = 3000;
const SLOW_MS = 5000; // the free server sleeps when idle; waking it takes up to a minute

const STAGE_LABELS: Record<string, string> = {
  resolve: "Checking the video",
  perceive: "Watching the lecture",
  validate: "Checking what was seen",
  align: "Matching speech to the slides",
  chunk: "Splitting into parts",
  extract: "Writing notes",
  merge: "Removing repetition",
  verify: "Fact-checking against the lecture",
  render: "Formatting",
};

export const isActive = (job?: Job) => job?.status === "queued" || job?.status === "running";

export function useJob(jobId: string | undefined) {
  return useQuery({
    queryKey: ["job", jobId],
    enabled: Boolean(jobId),
    queryFn: () =>
      call(api.GET("/v1/jobs/{job_id}", { params: { path: { job_id: jobId ?? "" } } })),
    refetchInterval: (query) =>
      query.state.data === undefined || isActive(query.state.data) ? POLL_MS : false,
    // Keep retrying network errors (a sleeping server wakes up); a 404 won't fix itself.
    retry: (failures, error) =>
      !(error instanceof ApiError && error.status === 404) && failures < 20,
  });
}

/** Live progress of a notes job: a bar plus each step, the current one spinning. */
export function JobProgress({ jobId }: { jobId: string }) {
  const job = useJob(jobId);
  const slow = useSlow(job.isFetching && job.data === undefined);

  if (job.data === undefined) {
    return (
      <div className="card progress-card">
        {job.isError ? <p className="alert error">{job.error.message}</p> : <p>Loading…</p>}
        {slow && !job.isError && (
          <p className="alert warn">
            The server is waking up (free hosting sleeps when idle). This can take up to a minute.
          </p>
        )}
      </div>
    );
  }

  const data = job.data;
  const finished = data.stages.filter((s) => s.state === "done" || s.state === "cached").length;
  const current = data.stages.find((s) => s.state === "pending")?.stage;
  const pct = Math.round((finished / Math.max(1, data.stages.length)) * 100);

  return (
    <div className="card progress-card">
      <h2 style={{ margin: 0, fontSize: "1.15rem" }}>{headline(data)}</h2>
      <p className="muted" style={{ margin: "4px 0 0", fontSize: 14 }}>
        A 10-minute lecture usually takes 2–4 minutes. You can leave this page — the lesson keeps
        processing.
      </p>
      {data.status === "failed" ? (
        <p className="alert error" role="alert" style={{ marginTop: 16 }}>
          {data.error}
        </p>
      ) : (
        <div className="progress-bar" aria-label={`${pct}% done`}>
          <span style={{ width: `${Math.max(pct, 4)}%` }} />
        </div>
      )}
      {data.status === "queued" && data.error_type && (
        <p className="alert warn">
          Waiting to retry ({data.error_type}). It continues automatically.
        </p>
      )}
      <ol className="stage-list">
        {data.stages.map((s) => {
          const state =
            s.state === "failed"
              ? "failed"
              : s.state === "pending"
                ? s.stage === current && data.status === "running"
                  ? "current"
                  : ""
                : "done";
          return (
            <li key={s.stage} className={state}>
              <span className="stage-icon">
                {state === "done" && <CheckIcon size={12} />}
                {state === "failed" && <XIcon size={12} />}
              </span>
              {STAGE_LABELS[s.stage] ?? s.stage}
              {s.state === "cached" && <span className="badge">reused</span>}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function headline(job: Job): string {
  switch (job.status) {
    case "queued":
      return "Waiting in line…";
    case "running":
      return "Making your notes…";
    case "succeeded":
      return "Done";
    case "failed":
      return "Couldn’t make notes for this video";
  }
}

/** True once `active` has stayed true for SLOW_MS. */
function useSlow(active: boolean): boolean {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    if (!active) {
      setSlow(false);
      return;
    }
    const timer = setTimeout(() => setSlow(true), SLOW_MS);
    return () => clearTimeout(timer);
  }, [active]);
  return slow;
}
