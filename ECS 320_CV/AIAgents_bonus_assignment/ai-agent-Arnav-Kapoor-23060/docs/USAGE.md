# Usage Guide

## Quick Start

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd agent-cv
   ```
2. Create a Python virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Generate exam paper, solutions, and syllabus:
   ```bash
   python src/generate_cv_paper.py --all
   ```
4. Use CLI options for custom generation:
   ```bash
   python src/generate_cv_paper.py --paper --objective 3 --subjective 2
   python src/generate_cv_paper.py --solutions
   python src/generate_cv_paper.py --syllabus
   python src/generate_cv_paper.py --llm "Generate a master's level MCQ on Vision Transformers" --api_key <your-openai-key>
   ```
5. Start the web UI:
   ```bash
   python scripts/webui.py
   ```

## Output Files
- `ECS320_CV_Endsem_Paper.txt`: Generated exam paper
- `ECS320_CV_Endsem_Paper_Solutions.txt`: Solution key
- `ECS320_CV_Syllabus.txt`: Syllabus

## Testing
Run all tests:
```bash
python -m unittest discover tests
```

## Continuous Integration
All tests run automatically on GitHub Actions for every push and pull request.
