"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, type Notes, toApiError, type Unit, type UnitEdit } from "@/lib/api";

const lines = (text: string) =>
  text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

/** One form per unit. Only the content is editable — where it was taught and what it cites
 * stay as extracted. Saving re-renders the notes on the server (no reprocessing). */
export function EditUnits({ notes }: { notes: Notes }) {
  return (
    <div className="edit-units">
      <p className="muted">
        Fix anything the notes got wrong. Timestamps and sources stay as extracted.
      </p>
      {notes.units.map((u) => (
        <UnitForm
          key={`${u.id}-${notes.revision}`}
          notesId={notes.id}
          unit={u}
          edited={notes.edited_units.includes(u.id)}
        />
      ))}
    </div>
  );
}

function UnitForm({ notesId, unit, edited }: { notesId: string; unit: Unit; edited: boolean }) {
  const queryClient = useQueryClient();
  const [concept, setConcept] = useState(unit.concept);
  const [section, setSection] = useState(unit.section ?? "");
  const [explanation, setExplanation] = useState(unit.explanation);
  const [examples, setExamples] = useState((unit.examples ?? []).join("\n"));
  const [boardText, setBoardText] = useState(unit.board_text ?? "");
  const [code, setCode] = useState((unit.code ?? []).map((c) => c.source));

  // Send only what changed, so an edit never overwrites fields the user didn't touch.
  const changes: UnitEdit = {};
  if (concept.trim() !== unit.concept) changes.concept = concept.trim();
  if (section.trim() !== (unit.section ?? "")) changes.section = section;
  if (explanation.trim() !== unit.explanation) changes.explanation = explanation.trim();
  if (lines(examples).join("\n") !== (unit.examples ?? []).join("\n")) {
    changes.examples = lines(examples);
  }
  if (boardText !== (unit.board_text ?? "")) changes.board_text = boardText;
  if (code.some((src, i) => src !== unit.code?.[i]?.source)) {
    changes.code = (unit.code ?? [])
      .map((c, i) => ({ ...c, source: code[i] ?? "" }))
      .filter((c) => c.source.trim());
  }
  const dirty = Object.keys(changes).length > 0;

  const save = useMutation({
    mutationFn: async () => {
      const { data, error, response } = await api.PATCH("/v1/notes/{notes_id}/units/{unit_id}", {
        params: { path: { notes_id: notesId, unit_id: unit.id } },
        body: changes,
      });
      if (error || !data) throw toApiError(error, response.status);
      return data;
    },
    onSuccess: (data) => queryClient.setQueryData(["notes", notesId], data),
    onError: (err) => {
      // Someone else saved first: load their version, the user re-applies on top of it.
      if ("status" in err && err.status === 409) {
        void queryClient.invalidateQueries({ queryKey: ["notes", notesId] });
      }
    },
  });

  return (
    <form
      className="card unit-form"
      onSubmit={(e) => {
        e.preventDefault();
        save.mutate();
      }}
    >
      <div className="unit-form-head">
        <input
          aria-label="Concept"
          value={concept}
          onChange={(e) => setConcept(e.target.value)}
          required
        />
        {edited && <span className="level">edited</span>}
      </div>
      <label>
        Section
        <input value={section} onChange={(e) => setSection(e.target.value)} />
      </label>
      <label>
        Explanation
        <textarea
          rows={4}
          value={explanation}
          onChange={(e) => setExplanation(e.target.value)}
          required
        />
      </label>
      <label>
        Examples (one per line)
        <textarea rows={3} value={examples} onChange={(e) => setExamples(e.target.value)} />
      </label>
      {(unit.board_text || boardText) && (
        <label>
          On the board
          <textarea
            className="mono"
            rows={3}
            value={boardText}
            onChange={(e) => setBoardText(e.target.value)}
          />
        </label>
      )}
      {code.map((src, i) => (
        <label key={i}>
          Code {code.length > 1 ? i + 1 : ""} (empty removes it)
          <textarea
            className="mono"
            rows={Math.min(12, src.split("\n").length + 1)}
            value={src}
            onChange={(e) => setCode(code.map((s, j) => (j === i ? e.target.value : s)))}
          />
        </label>
      ))}
      <div className="actions">
        <button type="submit" disabled={!dirty || save.isPending}>
          {save.isPending ? "Saving…" : "Save"}
        </button>
        {save.isError && <span className="alert error">{save.error.message}</span>}
        {save.isSuccess && !dirty && <span className="muted">Saved</span>}
      </div>
    </form>
  );
}
