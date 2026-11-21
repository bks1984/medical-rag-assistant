# Data

Put the medical manual PDF here as `medical_diagnosis_manual.pdf`. You can also point `MANUAL_PDF` in `.env` at another location.

The project was built on **The Merck Manual of Diagnosis & Therapy (19th edition)**. It has over 4,000 pages in 23 sections, which the pipeline splits into about 8,250 chunks. The PDF is copyrighted, so it is git-ignored and not part of this repository. Use your licensed copy, for example the one supplied with the course.

Any text-based medical PDF works. To use a different manual, set `MANUAL_NAME` in `.env` so citations show the right title, then run `python scripts/ingest.py` again.
