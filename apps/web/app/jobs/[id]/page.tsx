"use client";

import { useParams, useRouter } from "next/navigation";
import { useEffect } from "react";

import { JobProgress, useJob } from "@/components/job-progress";

/** Progress of a job by id (older links); opens its notes when done. */
export default function JobPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const notesId = useJob(id).data?.notes_id;

  useEffect(() => {
    if (notesId) router.push(`/notes/${notesId}`);
  }, [notesId, router]);

  return <JobProgress jobId={id} />;
}
