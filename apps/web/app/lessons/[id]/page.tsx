"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { useMe } from "@/components/auth";
import { NotesFeedback } from "@/components/feedback";
import {
  BookIcon,
  CopyIcon,
  DownloadIcon,
  FolderIcon,
  RefreshIcon,
  SparkIcon,
  YoutubeIcon,
} from "@/components/icons";
import { JobProgress, useJob } from "@/components/job-progress";
import { NotesView } from "@/components/notes-view";
import { Practice } from "@/components/practice";
import { InlineInput } from "@/components/sidebar";
import { api, call, exportUrl, formatTime } from "@/lib/api";
import {
  CHAPTERS_KEY,
  findLesson,
  useChapters,
  useRetryLesson,
  useUpdateLesson,
} from "@/lib/library";

type Tab = "notes" | "practice";

export default function LessonPage() {
  const { id } = useParams<{ id: string }>();
  const qc = useQueryClient();
  const [tab, setTab] = useState<Tab>("notes");
  const [renaming, setRenaming] = useState(false);
  const [copied, setCopied] = useState(false);
  const chapters = useChapters();
  const update = useUpdateLesson();
  const retry = useRetryLesson();
  const me = useMe().data;

  const lesson = useQuery({
    queryKey: ["lesson", id],
    queryFn: () =>
      call(api.GET("/v1/lessons/{lesson_id}", { params: { path: { lesson_id: id } } })),
  });
  const job = useJob(lesson.data?.notes_id ? undefined : lesson.data?.job_id);

  // When the job finishes, reload the lesson (it now has notes and the video's title).
  const jobStatus = job.data?.status;
  useEffect(() => {
    if (jobStatus === "succeeded" || jobStatus === "failed") {
      void qc.invalidateQueries({ queryKey: ["lesson", id] });
      void qc.invalidateQueries({ queryKey: CHAPTERS_KEY });
    }
  }, [jobStatus, id, qc]);

  const notesId = lesson.data?.notes_id;
  const notes = useQuery({
    queryKey: ["notes", notesId],
    enabled: Boolean(notesId),
    queryFn: () =>
      call(api.GET("/v1/notes/{notes_id}", { params: { path: { notes_id: notesId ?? "" } } })),
  });

  if (lesson.isError) {
    return (
      <div className="card empty">
        <h3>Lesson not found</h3>
        <p>
          It may have been deleted. <Link href="/">Add a new lesson</Link>
        </p>
      </div>
    );
  }
  if (!lesson.data) return <p className="muted">Loading…</p>;

  const data = lesson.data;
  const chapter = findLesson(chapters.data, data.id)?.chapter;
  const video = notes.data?.video;

  const copy = async () => {
    if (!notes.data) return;
    await navigator.clipboard.writeText(notes.data.markdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <>
      <header className="lesson-head">
        <div className="lesson-head-main">
          <div className="crumbs">
            <FolderIcon size={14} />
            {chapter?.name ?? "Chapter"}
            <span>›</span>
            <span>Lesson</span>
          </div>
          <h1 className="lesson-title">
            {renaming ? (
              <InlineInput
                initial={data.title}
                onDone={(title) => {
                  setRenaming(false);
                  if (title && title !== data.title) update.mutate({ id: data.id, title });
                }}
              />
            ) : (
              <button type="button" title="Click to rename" onClick={() => setRenaming(true)}>
                {data.title}
              </button>
            )}
          </h1>
          <div className="meta">
            <a href={data.url} target="_blank" rel="noopener noreferrer">
              <YoutubeIcon size={14} style={{ verticalAlign: "-2px", color: "#ff0033" }} />{" "}
              {video?.channel ?? "Watch on YouTube"}
            </a>
            {video && <span>{formatTime(video.duration_s)} lecture</span>}
            {notes.data && <span>{notes.data.units.length} notes</span>}
            {data.custom_title && video && (
              <span title="Original video title">“{video.title}”</span>
            )}
          </div>
        </div>
        {notes.data && (
          <div className="toolbar">
            <button type="button" className="btn ghost sm" onClick={copy}>
              <CopyIcon size={14} /> {copied ? "Copied" : "Copy Markdown"}
            </button>
            <a className="btn ghost sm" href={exportUrl(notes.data.id)} download>
              <DownloadIcon size={14} /> Download
            </a>
          </div>
        )}
      </header>

      {!notesId && data.status !== "failed" && <JobProgress jobId={data.job_id} />}

      {data.status === "failed" && !notesId && (
        <>
          <JobProgress jobId={data.job_id} />
          <div style={{ marginTop: 14 }}>
            <button
              type="button"
              className="btn"
              disabled={retry.isPending}
              onClick={() => retry.mutate(data.id)}
            >
              <RefreshIcon /> Try again
            </button>
            {retry.isError && <p className="alert error">{retry.error.message}</p>}
          </div>
        </>
      )}

      {notes.data && (
        <>
          <div className="tabs" role="tablist">
            <button
              type="button"
              role="tab"
              className="tab"
              aria-selected={tab === "notes"}
              onClick={() => setTab("notes")}
            >
              <BookIcon size={15} /> Notes
            </button>
            <button
              type="button"
              role="tab"
              className="tab"
              aria-selected={tab === "practice"}
              onClick={() => setTab("practice")}
            >
              <SparkIcon size={15} /> Practice
            </button>
          </div>
          {tab === "notes" ? (
            <>
              <NotesView notes={notes.data} canEdit={Boolean(me?.is_owner)} />
              <NotesFeedback notesId={notes.data.id} />
            </>
          ) : (
            <Practice notesId={notes.data.id} />
          )}
        </>
      )}
      {notesId && notes.isError && <p className="alert error">{notes.error.message}</p>}
    </>
  );
}
