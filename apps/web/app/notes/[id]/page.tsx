"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { api, exportUrl, toApiError } from "@/lib/api";

import { EditUnits } from "./edit-units";
import { PracticeQuestions } from "./practice";

export default function NotesPage() {
  const { id } = useParams<{ id: string }>();
  const [copied, setCopied] = useState(false);
  const [editing, setEditing] = useState(false);

  const notes = useQuery({
    queryKey: ["notes", id],
    queryFn: async () => {
      const { data, error, response } = await api.GET("/v1/notes/{notes_id}", {
        params: { path: { notes_id: id } },
      });
      if (error || !data) throw toApiError(error, response.status);
      return data;
    },
  });

  if (notes.isError) return <p className="alert error">{notes.error.message}</p>;
  if (!notes.data) return <p>Loading…</p>;

  const copy = async () => {
    await navigator.clipboard.writeText(notes.data.markdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <article>
      <div className="actions">
        <button type="button" onClick={copy}>
          {copied ? "Copied" : "Copy Markdown"}
        </button>
        <a className="button secondary" href={exportUrl(notes.data.id)} download>
          Download .md
        </a>
        {notes.data.units.length > 0 && (
          <button type="button" className="button secondary" onClick={() => setEditing(!editing)}>
            {editing ? "Done editing" : "Edit notes"}
          </button>
        )}
        <Link href="/" className="button secondary">
          New video
        </Link>
      </div>
      {editing ? (
        <EditUnits notes={notes.data} />
      ) : (
        <div className="card notes">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              // Timestamp links go to YouTube: open them next to the notes.
              a: ({ href, children }) => (
                <a href={href} target="_blank" rel="noopener noreferrer">
                  {children}
                </a>
              ),
            }}
          >
            {notes.data.markdown}
          </ReactMarkdown>
        </div>
      )}
      {notes.data.units.length > 0 && <PracticeQuestions notesId={notes.data.id} />}
    </article>
  );
}
