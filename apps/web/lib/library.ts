// Library data (chapters → lessons) and the mutations the sidebar and pages share.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, call, callEmpty, type Chapter, type Lesson, type SubjectPreset } from "./api";

export const CHAPTERS_KEY = ["chapters"] as const;

const isActive = (lesson: Lesson) => lesson.status === "queued" || lesson.status === "running";

export function useChapters() {
  return useQuery({
    queryKey: CHAPTERS_KEY,
    queryFn: () => call(api.GET("/v1/chapters")),
    // Keep lesson status dots live while anything is being processed.
    refetchInterval: (query) =>
      query.state.data?.some((c) => c.lessons.some(isActive)) ? 4000 : false,
  });
}

export function findLesson(chapters: Chapter[] | undefined, id: string) {
  for (const chapter of chapters ?? []) {
    const lesson = chapter.lessons.find((l) => l.id === id);
    if (lesson) return { chapter, lesson };
  }
  return undefined;
}

function useInvalidate() {
  const qc = useQueryClient();
  return () => qc.invalidateQueries({ queryKey: CHAPTERS_KEY });
}

export function useCreateChapter() {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (name: string) => call(api.POST("/v1/chapters", { body: { name } })),
    onSuccess: invalidate,
  });
}

export function useRenameChapter() {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) =>
      call(
        api.PATCH("/v1/chapters/{chapter_id}", {
          params: { path: { chapter_id: id } },
          body: { name },
        }),
      ),
    onSuccess: invalidate,
  });
}

export function useDeleteChapter() {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (id: string) =>
      callEmpty(api.DELETE("/v1/chapters/{chapter_id}", { params: { path: { chapter_id: id } } })),
    onSuccess: invalidate,
  });
}

export function useAddLesson() {
  const qc = useQueryClient();
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (args: { chapterId: string; url: string; preset: SubjectPreset; title?: string }) =>
      call(
        api.POST("/v1/chapters/{chapter_id}/lessons", {
          params: { path: { chapter_id: args.chapterId } },
          body: { url: args.url, subject_preset: args.preset, title: args.title || null },
        }),
      ),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["me"] });
      return invalidate();
    },
  });
}

export function useUpdateLesson() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...body }: { id: string; title?: string | null; chapter_id?: string }) =>
      call(api.PATCH("/v1/lessons/{lesson_id}", { params: { path: { lesson_id: id } }, body })),
    onSuccess: (lesson) => {
      qc.setQueryData(["lesson", lesson.id], lesson);
      return qc.invalidateQueries({ queryKey: CHAPTERS_KEY });
    },
  });
}

export function useDeleteLesson() {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (id: string) =>
      callEmpty(api.DELETE("/v1/lessons/{lesson_id}", { params: { path: { lesson_id: id } } })),
    onSuccess: invalidate,
  });
}

export function useRetryLesson() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      call(api.POST("/v1/lessons/{lesson_id}/retry", { params: { path: { lesson_id: id } } })),
    onSuccess: (lesson) => {
      qc.setQueryData(["lesson", lesson.id], lesson);
      return qc.invalidateQueries({ queryKey: CHAPTERS_KEY });
    },
  });
}
