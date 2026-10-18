"use client";

import { useQuery } from "@tanstack/react-query";
import { useParams } from "next/navigation";

import { NotesFeedback } from "@/components/feedback";
import { NotesView } from "@/components/notes-view";
import { Practice } from "@/components/practice";
import { api, call } from "@/lib/api";

/** Notes by id (older links, and notes not added to a chapter). */
export default function NotesPage() {
  const { id } = useParams<{ id: string }>();
  const notes = useQuery({
    queryKey: ["notes", id],
    queryFn: () => call(api.GET("/v1/notes/{notes_id}", { params: { path: { notes_id: id } } })),
  });

  if (notes.isError) return <p className="alert error">{notes.error.message}</p>;
  if (!notes.data) return <p className="muted">Loading…</p>;

  return (
    <>
      <header className="lesson-head">
        <h1 className="lesson-title">{notes.data.title}</h1>
      </header>
      <NotesView notes={notes.data} />
      <NotesFeedback notesId={notes.data.id} />
      <h2 className="section-title" style={{ marginTop: 40 }}>
        Practice
      </h2>
      <Practice notesId={notes.data.id} />
    </>
  );
}
