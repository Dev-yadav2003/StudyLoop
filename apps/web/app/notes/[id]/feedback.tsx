"use client";

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import { api, toApiError } from "@/lib/api";

/** "Were these notes useful?" — 👍 sends at once; 👎 asks what was wrong first. */
export function NotesFeedback({ notesId }: { notesId: string }) {
  const [rating, setRating] = useState<1 | -1 | null>(null);
  const [reason, setReason] = useState("");

  const send = useMutation({
    mutationFn: async (body: { rating: 1 | -1; reason?: string }) => {
      const { data, error, response } = await api.POST("/v1/notes/{notes_id}/feedback", {
        params: { path: { notes_id: notesId } },
        body,
      });
      if (error || !data) throw toApiError(error, response.status);
      return data;
    },
  });

  if (send.isSuccess) {
    return <p className="card feedback muted">Thanks — your feedback helps improve the notes.</p>;
  }

  return (
    <section className="card feedback">
      <div className="actions">
        <span>Were these notes useful?</span>
        <button
          type="button"
          className={rating === 1 ? "" : "button secondary"}
          aria-pressed={rating === 1}
          disabled={send.isPending}
          onClick={() => {
            setRating(1);
            send.mutate({ rating: 1 });
          }}
        >
          👍 Yes
        </button>
        <button
          type="button"
          className={rating === -1 ? "" : "button secondary"}
          aria-pressed={rating === -1}
          disabled={send.isPending}
          onClick={() => setRating(-1)}
        >
          👎 No
        </button>
      </div>
      {rating === -1 && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send.mutate({ rating: -1, reason });
          }}
        >
          <label>
            What was wrong? (optional)
            <textarea
              rows={3}
              maxLength={1000}
              value={reason}
              placeholder="e.g. a concept is missing, an explanation is wrong, timestamps are off…"
              onChange={(e) => setReason(e.target.value)}
            />
          </label>
          <button type="submit" disabled={send.isPending}>
            {send.isPending ? "Sending…" : "Send"}
          </button>
        </form>
      )}
      {send.isError && <p className="alert error">{send.error.message}</p>}
    </section>
  );
}
