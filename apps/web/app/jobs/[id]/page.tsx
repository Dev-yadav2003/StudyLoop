"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { ApiError, api, toApiError, type Job, type StageProgress } from "@/lib/api";

const POLL_MS = 3000;
const SLOW_MS = 5000; // the free server sleeps when idle; waking it takes up to a minute

const STAGE_LABELS: Record<string, string> = {
  resolve: "Checking the video",
  perceive: "Watching the video",
  validate: "Checking what was seen",
  align: "Matching speech to the board",
  chunk: "Splitting into sections",
  extract: "Writing notes",
  merge: "Removing duplicates",
  verify: "Fact-checking against the lecture",
  render: "Formatting",
};

const STATE_LABELS: Record<StageProgress["state"], string> = {
  pending: "waiting",
  done: "done",
  cached: "done (reused)",
  failed: "failed",
};

const isActive = (job?: Job) => job?.status === "queued" || job?.status === "running";

export default function JobPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();

  const job = useQuery({
    queryKey: ["job", id],
    queryFn: async () => {
      const { data, error, response } = await api.GET("/v1/jobs/{job_id}", {
        params: { path: { job_id: id } },
      });
      if (error || !data) throw toApiError(error, response.status);
      return data;
    },
    refetchInterval: (query) =>
      query.state.data === undefined || isActive(query.state.data) ? POLL_MS : false,
    // Keep retrying network errors (a sleeping server wakes up); a 404 won't fix itself.
    retry: (failures, error) => !(error instanceof ApiError && error.status === 404) && failures < 20,
  });

  const slow = useSlow(job.isFetching && job.data === undefined);
  const notesId = job.data?.notes_id;

  useEffect(() => {
    if (notesId) router.push(`/notes/${notesId}`);
  }, [notesId, router]);

  if (job.data === undefined) {
    return (
      <div className="card">
        {job.isError && job.error.message ? (
          <p className="alert error">{job.error.message}</p>
        ) : (
          <p>Loading…</p>
        )}
        {slow && !job.isError && (
          <p className="alert warn">
            The server is waking up (free hosting sleeps when idle). This can take up to a minute.
          </p>
        )}
      </div>
    );
  }

  const data = job.data;
  return (
    <div className="card">
      <h1>{headline(data)}</h1>
      <p className="muted">{data.url}</p>
      {data.status === "failed" && (
        <p className="alert error" role="alert">
          {data.error}
        </p>
      )}
      {data.status === "queued" && data.error_type && (
        <p className="alert warn">
          Waiting to retry ({data.error_type}). It continues automatically.
        </p>
      )}
      <ol className="stages">
        {data.stages.map((s) => (
          <li key={s.stage}>
            <span>{STAGE_LABELS[s.stage] ?? s.stage}</span>
            <span className={`state-${s.state}`}>{STATE_LABELS[s.state]}</span>
          </li>
        ))}
      </ol>
      {data.status === "failed" && (
        <div className="actions">
          <Link href="/" className="button secondary">
            Try another video
          </Link>
        </div>
      )}
    </div>
  );
}

function headline(job: Job): string {
  switch (job.status) {
    case "queued":
      return "Queued…";
    case "running":
      return "Working on your notes…";
    case "succeeded":
      return "Done — opening your notes";
    case "failed":
      return "Couldn't make notes for this video";
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
