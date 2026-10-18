"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { ApiError, api, call, type Unit, type UnitEdit } from "@/lib/api";

import { CheckIcon } from "./icons";

const lines = (text: string) =>
  text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

/** Edit one note in place. Only content is editable — where it was taught and what it cites
 * stay as extracted. Saving re-renders the notes on the server (no reprocessing, no AI). */
export function UnitEditor({
  notesId,
  unit,
  onClose,
}: {
  notesId: string;
  unit: Unit;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [concept, setConcept] = useState(unit.concept);
  const [section, setSection] = useState(unit.section ?? "");
  const [explanation, setExplanation] = useState(unit.explanation);
  const [examples, setExamples] = useState((unit.examples ?? []).join("\n"));
  const [code, setCode] = useState((unit.code ?? []).map((c) => c.source));

  // Send only what changed, so an edit never overwrites fields the user didn't touch.
  const changes: UnitEdit = {};
  if (concept.trim() !== unit.concept) changes.concept = concept.trim();
  if (section.trim() !== (unit.section ?? "")) changes.section = section;
  if (explanation.trim() !== unit.explanation) changes.explanation = explanation.trim();
  if (lines(examples).join("\n") !== (unit.examples ?? []).join("\n")) {
    changes.examples = lines(examples);
  }
  if (code.some((src, i) => src !== unit.code?.[i]?.source)) {
    changes.code = (unit.code ?? [])
      .map((c, i) => ({ ...c, source: code[i] ?? "" }))
      .filter((c) => c.source.trim());
  }
  const dirty = Object.keys(changes).length > 0;

  const save = useMutation({
    mutationFn: () =>
      call(
        api.PATCH("/v1/notes/{notes_id}/units/{unit_id}", {
          params: { path: { notes_id: notesId, unit_id: unit.id } },
          body: changes,
        }),
      ),
    onSuccess: (notes) => {
      qc.setQueryData(["notes", notesId], notes);
      onClose();
    },
    onError: (err) => {
      // Someone else saved first: load their version; the user re-applies on top of it.
      if (err instanceof ApiError && err.status === 409) {
        void qc.invalidateQueries({ queryKey: ["notes", notesId] });
      }
    },
  });

  return (
    <form
      className="editor"
      onSubmit={(e) => {
        e.preventDefault();
        save.mutate();
      }}
      onKeyDown={(e) => {
        if (e.key === "Escape") onClose();
      }}
    >
      <div className="editor-head">
        <span>Editing this note</span>
        <span style={{ fontWeight: 400 }}>Timestamps and sources stay as they are</span>
      </div>

      <div className="editor-grid">
        <label className="field">
          Title
          <input
            className="input"
            value={concept}
            onChange={(e) => setConcept(e.target.value)}
            required
            autoFocus
          />
        </label>
        <label className="field">
          Section
          <input className="input" value={section} onChange={(e) => setSection(e.target.value)} />
        </label>
      </div>

      <label className="field">
        <span>
          Note <span className="hint">— Markdown: **bold**, “- ” bullets, “1. ” steps</span>
        </span>
        <textarea
          className="textarea"
          rows={Math.min(16, Math.max(5, explanation.split("\n").length + 2))}
          value={explanation}
          onChange={(e) => setExplanation(e.target.value)}
          required
        />
      </label>

      <label className="field">
        <span>
          Examples <span className="hint">— one per line</span>
        </span>
        <textarea
          className="textarea"
          rows={Math.max(2, lines(examples).length + 1)}
          value={examples}
          onChange={(e) => setExamples(e.target.value)}
        />
      </label>

      {code.map((src, i) => (
        <label className="field" key={i}>
          <span>
            Code {code.length > 1 ? i + 1 : ""} <span className="hint">— empty removes it</span>
          </span>
          <textarea
            className="textarea mono"
            rows={Math.min(14, src.split("\n").length + 1)}
            value={src}
            onChange={(e) => setCode(code.map((s, j) => (j === i ? e.target.value : s)))}
          />
        </label>
      ))}

      <div className="editor-actions">
        {save.isError && <span className="alert error">{save.error.message}</span>}
        <button type="button" className="btn ghost sm" onClick={onClose}>
          Cancel
        </button>
        <button type="submit" className="btn sm" disabled={!dirty || save.isPending}>
          <CheckIcon size={14} /> {save.isPending ? "Saving…" : "Save changes"}
        </button>
      </div>
    </form>
  );
}
