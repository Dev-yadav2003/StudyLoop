"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { formatTime, youtubeAt, type Notes, type Unit } from "@/lib/api";

import { PencilIcon, PlayIcon } from "./icons";
import { UnitEditor } from "./unit-editor";

type Section = { title: string; units: Unit[] };

/** Same order as the Markdown export (render stage): sections by when they are first taught,
 * concepts by time within a section. */
export function sectionsOf(units: Unit[]): Section[] {
  const sorted = [...units].sort(
    (a, b) => (a.spans[0]?.start_s ?? 0) - (b.spans[0]?.start_s ?? 0) || a.id.localeCompare(b.id),
  );
  const sections = new Map<string, Unit[]>();
  for (const unit of sorted) {
    const title = (unit.section ?? "").replace(/\s+/g, " ").trim() || "Notes";
    sections.set(title, [...(sections.get(title) ?? []), unit]);
  }
  return [...sections].map(([title, members]) => ({ title, units: members }));
}

const anchor = (id: string) => `u-${id.replace(/[^a-zA-Z0-9_-]/g, "-")}`;

export function Markdown({ children }: { children: string }) {
  return (
    <div className="prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children: text }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {text}
            </a>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

export function NotesView({ notes }: { notes: Notes }) {
  const sections = sectionsOf(notes.units);

  if (notes.units.length === 0) {
    return (
      <div className="card empty">
        <h3>No notes could be extracted</h3>
        <p>The video may have no teaching content we could read.</p>
      </div>
    );
  }

  return (
    <div className="notes-layout">
      <div>
        {sections.map((section, i) => (
          <section key={section.title} className="notes-section">
            <h2 className="section-title" id={`s-${i + 1}`}>
              <span className="section-num">{i + 1}</span>
              {section.title}
            </h2>
            {section.units.map((unit) => (
              <UnitCard
                key={unit.id}
                notesId={notes.id}
                videoId={notes.video.video_id}
                unit={unit}
                edited={notes.edited_units.includes(unit.id)}
              />
            ))}
          </section>
        ))}
      </div>
      <nav className="outline" aria-label="On this page">
        <h4>On this page</h4>
        <ol>
          {sections.map((section, i) => (
            <li key={section.title}>
              <a href={`#s-${i + 1}`}>
                {i + 1}. {section.title}
              </a>
              <ol className="sub">
                {section.units.map((u) => (
                  <li key={u.id}>
                    <a href={`#${anchor(u.id)}`}>{u.concept}</a>
                  </li>
                ))}
              </ol>
            </li>
          ))}
        </ol>
      </nav>
    </div>
  );
}

function UnitCard({
  notesId,
  videoId,
  unit,
  edited,
}: {
  notesId: string;
  videoId: string;
  unit: Unit;
  edited: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [primary, ...others] = unit.spans;
  const examples = unit.examples ?? [];

  return (
    <article className="unit" id={anchor(unit.id)}>
      <div className="unit-head">
        <h3 className="unit-title">
          {unit.concept}
          {edited && <span className="badge accent edited-tag">edited</span>}
        </h3>
        {!editing && (
          <div className="unit-tools">
            <button
              type="button"
              className="btn quiet sm"
              title="Edit this note"
              aria-label={`Edit ${unit.concept}`}
              onClick={() => setEditing(true)}
            >
              <PencilIcon size={14} /> Edit
            </button>
          </div>
        )}
      </div>

      <div className="times">
        {primary && (
          <a
            className="time primary"
            href={youtubeAt(primary.video_id || videoId, primary.start_s)}
            target="_blank"
            rel="noopener noreferrer"
            title="Watch where this is taught"
          >
            <PlayIcon size={10} /> {formatTime(primary.start_s)}
          </a>
        )}
        {others.length > 0 && <span className="also">also at</span>}
        {others.map((s, i) => (
          <a
            key={`${i}-${s.start_s}`}
            className="time"
            href={youtubeAt(s.video_id || videoId, s.start_s)}
            target="_blank"
            rel="noopener noreferrer"
            title="Also mentioned here"
          >
            {formatTime(s.start_s)}
          </a>
        ))}
      </div>

      {editing ? (
        <UnitEditor notesId={notesId} unit={unit} onClose={() => setEditing(false)} />
      ) : (
        <>
          <Markdown>{unit.explanation}</Markdown>
          {(unit.code ?? []).map((block, i) => (
            <div className="code-block" key={i}>
              {block.language && <span className="lang">{block.language}</span>}
              <pre>
                <code>{block.source}</code>
              </pre>
            </div>
          ))}
          {examples.length > 0 && (
            <div className="examples">
              <h5>{examples.length === 1 ? "Example" : "Examples"}</h5>
              {examples.length === 1 ? (
                <Markdown>{examples[0] ?? ""}</Markdown>
              ) : (
                <ul>
                  {examples.map((e) => (
                    <li key={e}>
                      <Markdown>{e}</Markdown>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </>
      )}
    </article>
  );
}
