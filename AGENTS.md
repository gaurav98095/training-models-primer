# Training Models Primer — Agent Playbook

## Mission

This repository is a visual, module-by-module handbook for deep-learning training engineering, GPU systems, and LLM training. Every file under `docs/` should read like a polished chapter that a learner can study independently.

When a new module is added, process it completely: clean the source, improve the explanation, add useful visuals, extract practical code into a notebook, and validate the result.

## Repository layout

```text
docs/NN-topic.md                 Learner-facing chapter
docs/assets/module-NN/           Module-specific diagrams and images
notebooks/NN-topic.ipynb         Companion practical notebook
```

Keep the module number and topic slug consistent across the Markdown file, asset directory, and notebook.

## Module workflow

For every new or updated chapter:

1. Read the complete source before editing.
2. Identify the learning objective, concepts, equations, examples, code, and visuals.
3. Repair copied-content damage:
   - restore Python indentation and line breaks;
   - repair code fences and language identifiers;
   - fix Markdown tables and LaTeX delimiters;
   - reconnect captions to the correct figures;
   - replace missing or flattened visual placeholders.
4. Rewrite for clarity while preserving the source’s technical meaning.
5. Keep the chapter self-contained and suitable for someone learning the topic from fundamentals.
6. Create or update the matching notebook under `notebooks/`.
7. Add visual checkpoints throughout the chapter, not just one decorative banner.
8. Validate links, visuals, Markdown structure, notebook JSON, and Python syntax.

If the source file is empty or incomplete, report that instead of inventing a full chapter from the filename.

## Writing and layout rules

- Use a concise title such as `Module 01 — Anatomy of One LLM Training Step`.
- Do not use curriculum-count wording such as `Module 1 of 40` or `Module 01 of 40`.
- Avoid unnecessary references to specific future module numbers.
- Start with a short orientation and a clear central idea.
- Prefer short paragraphs, worked examples, tables, checklists, and diagrams over dense prose.
- Use numbered sections only when they represent a genuine learning progression.
- For debugging topics, use the pattern `change → observation → lesson`.
- Keep explanations close to the code or visual they explain.
- Use fenced code blocks with language identifiers such as `python`, `bash`, or `text`.
- End with a compact mental model, checklist, or knowledge check when useful.

## Visual learning rules

Visuals should help the learner see what changes, what flows, or what must be inspected.

- Put module assets in `docs/assets/module-NN/`.
- Use lowercase, descriptive, hyphenated filenames.
- Use SVG for exact technical diagrams: tensor shapes, data flow, state transitions, equations, timelines, and architecture.
- Use generated raster artwork for conceptual chapter headers or visual metaphors when it adds value.
- Keep essential facts in Markdown or SVG; never rely on generated artwork for exact technical labels.
- Add a short caption or nearby explanation for every diagram.
- Pair important visuals with a learner prompt such as “Pause and predict” or a before/after comparison.
- Use `<details>` for optional explanations, answer reveals, or deeper dives when the renderer supports it.
- Do not add decorative images that compete with the lesson.
- Do not describe a static image as interactive. Make it interactive-feeling through prompts, reveals, comparisons, and guided visual checkpoints.
- Verify every relative image reference from the Markdown file.

Example:

```markdown
![The five stages of one training step](assets/module-01/training-step-flow.svg)
```

## Notebook rules

Create one companion notebook per module:

```text
notebooks/NN-topic.ipynb
```

- Keep practical code in the Markdown chapter and copy it into notebook cells.
- Keep names, constants, tensor shapes, and logic aligned between the two files.
- Use explanatory Markdown cells before focused code cells.
- Include imports, seeds where useful, device selection, setup, the main experiment, diagnostics, and exercises.
- Make the notebook runnable from a clean kernel in top-to-bottom order.
- Avoid hidden state, undefined variables, and out-of-order cell dependencies.
- Use CPU-safe defaults and opportunistic GPU selection with `torch.cuda.is_available()`.
- Do not fabricate output cells or benchmark numbers.
- If dependencies are unavailable, validate syntax and clearly report that runtime execution was not possible.

## Validation checklist

Before handing back a module, check:

- Markdown code fences are balanced.
- All Python code blocks parse successfully.
- Notebook JSON is valid.
- Every notebook code cell parses successfully.
- Notebook cells are ordered for a clean-kernel run.
- Every Markdown image reference resolves.
- Every SVG asset parses correctly.
- Tables have consistent columns and readable headings.
- Equations use consistent delimiters.
- No `Module X of 40` or unnecessary future-module wording remains.
- No flattened indentation, broken code, placeholder chart labels, or detached captions remain.
- Runtime execution is attempted when dependencies are available.

## Change discipline

- Use `apply_patch` for text and structured-file edits.
- Preserve unrelated user work.
- Keep changes focused on the requested module and its companion assets.
- Do not overwrite or delete existing assets unless replacement is explicitly requested.
- Keep intentional source assets tracked. Do not add generated caches, model checkpoints, datasets, or secrets.

## Handoff format

Report briefly:

1. Markdown chapter updated.
2. Notebook created or updated.
3. Visuals added or repaired.
4. Validation completed.
5. Any runtime checks blocked by missing dependencies.
