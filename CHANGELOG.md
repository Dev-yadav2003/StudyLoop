# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-19

First public version.

### Added
- Notes from a YouTube lecture: speech and on-screen content (slides, whiteboard, code),
  organised in sections, each concept linked to the moment it's taught.
- Faithfulness checks: every note cites the lecture; unsupported claims are rewritten or
  dropped; quoted code and its output must appear on a slide.
- Library: chapters and lessons with rename, move, delete and retry.
- Practice questions (recall / application) generated only from the notes, with
  self-check.
- In-place note editing, Markdown copy and download, feedback (👍/👎).
- Sign-in by email link; owner accounts unlimited, guest accounts with 1 hour of video.
- LLM gateway with structured output, retries, API-key failover and model fallback.
- Content-addressed stage cache, Postgres job queue with crash recovery, Langfuse tracing.
- Evaluation suite: faithfulness, duplicates, section structure, question correctness,
  code compile/run/output checks; CI eval gate on labelled pull requests.
