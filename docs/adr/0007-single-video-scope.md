
## Context

The plan (§1, Phase 7) included playlists: expand a playlist link, process every video, and
merge notes across videos. The product owner decided the product is for single videos: when a
user gives a link, they want everything for *that* video only.

## Decision

- A link resolves to exactly one video. A watch link that also carries `list=` (a video opened
  from a playlist) means the video it points at.
- A bare `/playlist?list=…` link is rejected with `PlaylistNotSupported` (a subclass of
  `InvalidLink`) and the message "Playlists aren't supported. Open the video you want from the
  playlist and paste its link." No API call is made.
- Playlist expansion was **removed**, not disabled: `ResolvedVideos`, `SkippedVideo`, playlist
  fields on `VideoRef`/`VideoMeta`, the paginated `playlistItems` code and the `job_videos`
  table (migration 0003) are gone. `Pipeline.run` returns one `NotesDocument`, which has one
  `video`.
- Phase 7's merge stage now targets duplicates **within** a video (the same concept extracted
  from several chunks), which the first real run showed is the actual problem.
- `KnowledgeUnit.spans` stays a list: one concept can be taught at several moments of the
  same video, and merged units keep every timestamp.

## Consequences

- Simpler resolve stage (1 API unit per job), simpler domain and pipeline, fewer failure modes
  (no partial playlists, no per-video skip lists).
- Golden set: the planned "playlist of 3 related videos" case is dropped.
- Re-introducing playlists later would mean a new resolve output type and a cross-video merge;
  the content-addressed stage cache would still let per-video work be reused.
