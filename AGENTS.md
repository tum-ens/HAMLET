# AGENTS.md

Read [`.ai/context.md`](.ai/context.md) before touching anything in this repository.

It is the single orientation document for every AI tool working here — architecture, environment,
solver, determinism, branch rules, and the working rules that are not negotiable. This file is a
pointer so that Claude, Gemini and Codex-style agents all read the same text instead of three
copies that drift apart.

`AGENTS.md` is the cross-tool convention: OpenAI Codex, Mistral Vibe, Cursor and others look for
this filename. `CLAUDE.md` and `GEMINI.md` are the same pointer under the names their own tools
look for.

Do not add repository guidance here. Add it to `.ai/context.md`, and only if you have verified it.
