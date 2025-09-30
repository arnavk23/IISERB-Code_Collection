# System Architecture

## Overview
This project is organized for modularity, extensibility, and professional development.

### Main Components
- **src/generate_cv_paper.py**: Core logic for generating papers, solutions, syllabus, and LLM integration.
- **scripts/**: PDF export, architecture diagram, and web UI.
- **tests/**: Unit and CLI tests for all major features.
- **data/**: Output files and sample data.
- **docs/**: Documentation and guides.

### Flow
1. Instructor or user runs CLI/web UI.
2. Paper, solutions, and syllabus are generated using core logic.
3. LLM integration (OpenAI) can generate advanced questions.
4. All outputs are saved to `data/` or project root.
5. Tests ensure correctness and CI runs on every push.

### Extensibility
- Add new question types or sources in `src/generate_cv_paper.py`.
- Extend web UI in `scripts/webui.py`.
- Add more tests in `tests/`.
- Integrate other LLMs (e.g., HuggingFace) as needed.
