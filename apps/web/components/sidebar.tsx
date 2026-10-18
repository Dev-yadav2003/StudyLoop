"use client";

import Link from "next/link";
import { useQueryClient } from "@tanstack/react-query";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, type KeyboardEvent } from "react";

import type { Chapter, Lesson } from "@/lib/api";
import {
  useChapters,
  useCreateChapter,
  useDeleteChapter,
  useDeleteLesson,
  useRenameChapter,
  useUpdateLesson,
} from "@/lib/library";

import {
  BookIcon,
  ChevronIcon,
  FolderIcon,
  PencilIcon,
  PlusIcon,
  SearchIcon,
  TrashIcon,
} from "./icons";
import { ThemeToggle } from "./theme-toggle";
import { signOut, useMe, useSession } from "./auth";

/** Library navigation: chapters (folders) with their lessons, all renamable in place. */
export function Sidebar({ onNavigate }: { onNavigate: () => void }) {
  const chapters = useChapters();
  const createChapter = useCreateChapter();
  const [query, setQuery] = useState("");
  const [adding, setAdding] = useState(false);

  const q = query.trim().toLowerCase();
  const visible = (chapters.data ?? [])
    .map((c) => ({
      ...c,
      lessons: q ? c.lessons.filter((l) => l.title.toLowerCase().includes(q)) : c.lessons,
    }))
    .filter((c) => !q || c.name.toLowerCase().includes(q) || c.lessons.length > 0);

  return (
    <aside className="sidebar" aria-label="Library">
      <div className="sidebar-head">
        <Link href="/" className="brand" onClick={onNavigate}>
          <span className="brand-mark">
            <BookIcon size={17} />
          </span>
          StudyLoop
        </Link>
      </div>
      <div className="new-lesson">
        <Link href="/" className="btn block" onClick={onNavigate}>
          <PlusIcon /> New lesson
        </Link>
      </div>
      <label className="sidebar-search">
        <span className="sr-only">Search lessons</span>
        <SearchIcon size={15} />
        <input
          type="search"
          placeholder="Search lessons"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </label>

      <nav className="sidebar-scroll">
        <div className="sidebar-label">
          Chapters
          <button
            type="button"
            className="icon-btn"
            title="New chapter"
            aria-label="New chapter"
            onClick={() => setAdding(true)}
          >
            <PlusIcon size={15} />
          </button>
        </div>

        {adding && (
          <div className="tree-row">
            <FolderIcon />
            <InlineInput
              initial=""
              placeholder="Chapter name, e.g. Python"
              onDone={(name) => {
                setAdding(false);
                if (name) createChapter.mutate(name);
              }}
            />
          </div>
        )}

        {chapters.isError && <p className="empty-tree">Can't reach the server.</p>}
        {chapters.data?.length === 0 && !adding && (
          <p className="empty-tree">
            No chapters yet. Create one (e.g. “Python”) and add lessons to it.
          </p>
        )}
        {visible.map((chapter) => (
          <ChapterNode
            key={chapter.id}
            chapter={chapter}
            forceOpen={Boolean(q)}
            onNavigate={onNavigate}
          />
        ))}
      </nav>

      <Account />
    </aside>
  );
}

function ChapterNode({
  chapter,
  forceOpen,
  onNavigate,
}: {
  chapter: Chapter;
  forceOpen: boolean;
  onNavigate: () => void;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const containsCurrent = chapter.lessons.some((l) => pathname === `/lessons/${l.id}`);
  const [open, setOpen] = useState(true);
  const [renaming, setRenaming] = useState(false);
  const rename = useRenameChapter();
  const remove = useDeleteChapter();

  useEffect(() => {
    if (containsCurrent) setOpen(true);
  }, [containsCurrent]);

  const expanded = open || forceOpen;

  return (
    <div>
      <div
        className="tree-row"
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        onClick={() => !renaming && setOpen(!open)}
        onKeyDown={(e) => {
          if (!renaming && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            setOpen(!open);
          }
        }}
      >
        <ChevronIcon size={14} className={`chevron ${expanded ? "open" : ""}`} />
        <FolderIcon />
        {renaming ? (
          <InlineInput
            initial={chapter.name}
            onDone={(name) => {
              setRenaming(false);
              if (name && name !== chapter.name) rename.mutate({ id: chapter.id, name });
            }}
          />
        ) : (
          <>
            <span className="name">{chapter.name}</span>
            <span className="count">{chapter.lessons.length}</span>
            <span className="row-actions" onClick={(e) => e.stopPropagation()}>
              <button
                type="button"
                className="icon-btn"
                title="Add a lesson to this chapter"
                aria-label={`Add a lesson to ${chapter.name}`}
                onClick={() => {
                  router.push(`/?chapter=${chapter.id}`);
                  onNavigate();
                }}
              >
                <PlusIcon size={14} />
              </button>
              <button
                type="button"
                className="icon-btn"
                title="Rename"
                aria-label={`Rename ${chapter.name}`}
                onClick={() => setRenaming(true)}
              >
                <PencilIcon size={14} />
              </button>
              <button
                type="button"
                className="icon-btn danger"
                title="Delete chapter"
                aria-label={`Delete ${chapter.name}`}
                onClick={() => {
                  const n = chapter.lessons.length;
                  const what = n ? ` and its ${n} lesson${n === 1 ? "" : "s"}` : "";
                  if (confirm(`Delete “${chapter.name}”${what}?`)) {
                    remove.mutate(chapter.id, {
                      onSuccess: () => containsCurrent && router.push("/"),
                    });
                  }
                }}
              >
                <TrashIcon size={14} />
              </button>
            </span>
          </>
        )}
      </div>
      {expanded && (
        <div className="tree-children">
          {chapter.lessons.length === 0 && <p className="empty-tree">No lessons yet</p>}
          {chapter.lessons.map((lesson) => (
            <LessonNode key={lesson.id} lesson={lesson} onNavigate={onNavigate} />
          ))}
        </div>
      )}
    </div>
  );
}

function LessonNode({ lesson, onNavigate }: { lesson: Lesson; onNavigate: () => void }) {
  const pathname = usePathname();
  const router = useRouter();
  const active = pathname === `/lessons/${lesson.id}`;
  const [renaming, setRenaming] = useState(false);
  const update = useUpdateLesson();
  const remove = useDeleteLesson();

  if (renaming) {
    return (
      <div className="tree-row active">
        <span className={`dot ${lesson.status}`} />
        <InlineInput
          initial={lesson.title}
          onDone={(title) => {
            setRenaming(false);
            if (title && title !== lesson.title) update.mutate({ id: lesson.id, title });
          }}
        />
      </div>
    );
  }

  return (
    <Link
      href={`/lessons/${lesson.id}`}
      className={`tree-row ${active ? "active" : ""}`}
      onClick={onNavigate}
      title={lesson.title}
    >
      <span className={`dot ${lesson.status}`} aria-label={lesson.status} />
      <span className="name">{lesson.title}</span>
      <span
        className="row-actions"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
        }}
      >
        <button
          type="button"
          className="icon-btn"
          title="Rename"
          aria-label={`Rename ${lesson.title}`}
          onClick={() => setRenaming(true)}
        >
          <PencilIcon size={14} />
        </button>
        <button
          type="button"
          className="icon-btn danger"
          title="Delete lesson"
          aria-label={`Delete ${lesson.title}`}
          onClick={() => {
            if (confirm(`Delete the lesson “${lesson.title}”?`)) {
              remove.mutate(lesson.id, { onSuccess: () => active && router.push("/") });
            }
          }}
        >
          <TrashIcon size={14} />
        </button>
      </span>
    </Link>
  );
}

/** Text input that commits on Enter/blur and cancels on Escape (commits "" = cancel). */
export function InlineInput({
  initial,
  placeholder,
  onDone,
}: {
  initial: string;
  placeholder?: string;
  onDone: (value: string) => void;
}) {
  const [value, setValue] = useState(initial);
  const [done, setDone] = useState(false);
  const finish = (v: string) => {
    if (done) return;
    setDone(true);
    onDone(v.trim());
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") finish(value);
    if (e.key === "Escape") finish("");
  };
  return (
    <input
      autoFocus
      value={value}
      maxLength={200}
      placeholder={placeholder}
      onChange={(e) => setValue(e.target.value)}
      onKeyDown={onKey}
      onBlur={() => finish(value)}
      onClick={(e) => e.stopPropagation()}
    />
  );
}

/** Who is signed in, what they have left (guests), theme and sign-out. */
function Account() {
  const session = useSession();
  const me = useMe().data;
  const qc = useQueryClient();
  const limit = me?.video_minutes_limit;
  const pct = limit ? Math.min(100, ((me?.video_minutes_used ?? 0) / limit) * 100) : 0;

  return (
    <div className="sidebar-foot account">
      <div className="account-row">
        <span className="avatar">{(session?.user.email ?? "?").slice(0, 1).toUpperCase()}</span>
        <span className="account-email" title={session?.user.email}>
          {session?.user.email}
        </span>
        <ThemeToggle />
      </div>
      {me &&
        (me.is_owner ? (
          <span className="badge accent">Owner · unlimited</span>
        ) : (
          <div className="allowance" title="Free accounts: 1 hour of lecture video in total">
            <div className="allowance-bar">
              <span style={{ width: `${pct}%` }} />
            </div>
            <span>
              {Math.round(me.video_minutes_used)} of {limit} min of video used
            </span>
          </div>
        ))}
      <button type="button" className="btn quiet sm" onClick={() => signOut(() => qc.clear())}>
        Sign out
      </button>
    </div>
  );
}
