"use client";

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import { api, call } from "@/lib/api";

import { ThumbDownIcon, ThumbUpIcon } from "./icons";

/** "Were these notes useful?" — 👍 sends at once; 👎 asks what was wrong first. */
export function NotesFeedback({ notesId }: { notesId: string }) {
  const [down, setDown] = useState(false);
  const [reason, setReason] = useState("");

  const send = useMutation({
    mutationFn: (body: { rating: 1 | -1; reason?: string }) =>
      call(
        api.POST("/v1/notes/{notes_id}/feedback", {
          params: { path: { notes_id: notesId } },
          body,
        }),
      ),
  });

  if (send.isSuccess) {
    return <div className="feedback">Thanks — your feedback helps improve the notes.</div>;
  }

  return (
    <div className="feedback">
      <span>Were these notes useful?</span>
      <button
        type="button"
        className="btn ghost sm"
        disabled={send.isPending}
        onClick={() => send.mutate({ rating: 1 })}
      >
        <ThumbUpIcon size={14} /> Yes
      </button>
      <button
        type="button"
        className="btn ghost sm"
        aria-pressed={down}
        disabled={send.isPending}
        onClick={() => setDown(true)}
      >
        <ThumbDownIcon size={14} /> No
      </button>
      {down && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send.mutate({ rating: -1, reason });
          }}
        >
          <textarea
            className="textarea"
            rows={3}
            maxLength={1000}
            autoFocus
            value={reason}
            placeholder="What was wrong? e.g. a concept is missing, an explanation is wrong…"
            onChange={(e) => setReason(e.target.value)}
          />
          <div>
            <button type="submit" className="btn sm" disabled={send.isPending}>
              {send.isPending ? "Sending…" : "Send"}
            </button>
          </div>
        </form>
      )}
      {send.isError && <p className="alert error">{send.error.message}</p>}
    </div>
  );
}
