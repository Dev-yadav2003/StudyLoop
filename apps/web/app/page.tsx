"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState, type FormEvent } from "react";

import { BookIcon, CheckIcon, SparkIcon, YoutubeIcon } from "@/components/icons";
import type { SubjectPreset } from "@/lib/api";
import { useAddLesson, useChapters, useCreateChapter } from "@/lib/library";

const PRESETS: { value: SubjectPreset; label: string; hint: string }[] = [
  { value: "general", label: "General", hint: "Any lecture" },
  { value: "stem", label: "STEM", hint: "Formulas and code first" },
  { value: "theory", label: "Theory", hint: "Arguments and definitions first" },
];

const NEW_CHAPTER = "__new__";

export default function HomePage() {
  return (
    <Suspense>
      <Composer />
    </Suspense>
  );
}

function Composer() {
  const router = useRouter();
  const params = useSearchParams();
  const chapters = useChapters();
  const createChapter = useCreateChapter();
  const addLesson = useAddLesson();

  const [url, setUrl] = useState("");
  const [title, setTitle] = useState("");
  const [preset, setPreset] = useState<SubjectPreset>("general");
  const [chapterId, setChapterId] = useState<string>("");
  const [newChapter, setNewChapter] = useState("");

  // Default chapter: the one asked for in the URL (sidebar "+"), else the first one,
  // else a new one (there are none yet).
  useEffect(() => {
    if (!chapters.data || chapterId) return;
    const wanted = params.get("chapter");
    const first = chapters.data[0]?.id;
    setChapterId(
      wanted && chapters.data.some((c) => c.id === wanted) ? wanted : (first ?? NEW_CHAPTER),
    );
  }, [chapters.data, chapterId, params]);

  useEffect(() => {
    const wanted = params.get("chapter");
    if (wanted && chapters.data?.some((c) => c.id === wanted)) setChapterId(wanted);
  }, [params, chapters.data]);

  const creatingChapter = chapterId === NEW_CHAPTER;
  const busy = createChapter.isPending || addLesson.isPending;
  const error = createChapter.error ?? addLesson.error;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    let target = chapterId;
    if (creatingChapter) {
      const chapter = await createChapter.mutateAsync(newChapter.trim() || "My lessons");
      target = chapter.id;
      setChapterId(chapter.id);
    }
    const lesson = await addLesson.mutateAsync({
      chapterId: target,
      url: url.trim(),
      preset,
      title: title.trim() || undefined,
    });
    router.push(`/lessons/${lesson.id}`);
  };

  return (
    <div className="home">
      <h1 className="hero-title">What are you learning today?</h1>
      <p className="hero-sub">
        Paste a YouTube lecture. You get clean study notes with timestamps, practice questions, and
        a place for it in your chapters.
      </p>

      <form
        className="composer"
        onSubmit={(e) => {
          submit(e).catch(() => {
            // shown below via the mutations' error state
          });
        }}
      >
        <label className="composer-link">
          <YoutubeIcon size={22} className="yt" />
          <span className="sr-only">YouTube video link</span>
          <input
            type="url"
            required
            autoFocus
            placeholder="Paste a YouTube video link…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <button type="submit" className="btn" disabled={busy || !url.trim()}>
            {busy ? "Adding…" : "Make notes"}
          </button>
        </label>

        <div className={`composer-row ${creatingChapter ? "with-new" : ""}`}>
          <label className="field">
            Chapter
            <select
              className="select"
              value={chapterId}
              onChange={(e) => setChapterId(e.target.value)}
            >
              {chapters.data?.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
              <option value={NEW_CHAPTER}>+ New chapter…</option>
            </select>
          </label>
          {creatingChapter && (
            <label className="field">
              New chapter name
              <input
                className="input"
                placeholder="e.g. Python"
                maxLength={80}
                value={newChapter}
                onChange={(e) => setNewChapter(e.target.value)}
              />
            </label>
          )}
          <label className="field">
            <span>
              Lesson name <span className="hint">— optional, defaults to the video title</span>
            </span>
            <input
              className="input"
              placeholder="e.g. Lesson 3: Loops"
              maxLength={200}
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
          </label>
        </div>

        <div className="composer-foot" role="radiogroup" aria-label="Subject type">
          <span className="label">Lecture type</span>
          {PRESETS.map((p) => (
            <button
              key={p.value}
              type="button"
              className="chip"
              role="radio"
              aria-checked={preset === p.value}
              aria-pressed={preset === p.value}
              title={p.hint}
              onClick={() => setPreset(p.value)}
            >
              {preset === p.value && <CheckIcon size={13} />}
              {p.label}
            </button>
          ))}
          <span className="muted" style={{ fontSize: 13 }}>
            {PRESETS.find((p) => p.value === preset)?.hint}
          </span>
        </div>

        {error && (
          <p className="alert error" role="alert">
            {error.message}
          </p>
        )}
      </form>

      <div className="steps">
        <div className="step">
          <strong>
            <YoutubeIcon size={15} /> Paste one lecture
          </strong>
          <p>Any public video. Playlists aren’t supported — add lessons one by one.</p>
        </div>
        <div className="step">
          <strong>
            <BookIcon size={15} /> Read clean notes
          </strong>
          <p>Definitions, lists and code — each linked to the moment it’s taught.</p>
        </div>
        <div className="step">
          <strong>
            <SparkIcon size={15} /> Practise
          </strong>
          <p>Questions grounded in your notes, with answers you can reveal.</p>
        </div>
      </div>
    </div>
  );
}
