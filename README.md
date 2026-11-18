# AI Colorist
An AI tool that colour grades photos (and soon video) from plain-English commands, without repainting the picture.

## How it works
- brain.py: turns a typed command into grading settings (Gemini API)
- colorist.py: applies those settings with dials, curves, colour wheels and HSL
- Guardrails: skin-tone protection, automatic highlight/shadow safety, undo, and exportable recipes

## Status
Photos: working. Masking: next. Video and an Android app: planned.
