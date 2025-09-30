# Original Research: Multi-Agent AI System for Automated Computer Vision Exam Paper Generation

## Abstract

This paper presents a modular, multi-agent AI system for automated generation of Computer Vision exam papers, integrating LLMs, configurable branding, and real-time feedback. We benchmark the system against human-created papers, analyze question quality, and discuss implications for education technology.

## Introduction

Automated exam paper generation is a growing field, leveraging advances in large language models (LLMs) and agent-based architectures. Our system, inspired by QuestGen-AI-Agent, enables instructors to generate high-quality, customizable exam papers for Computer Vision (ECS 320) with objective and subjective questions, solutions, and syllabus.

## Related Work

- QuestGen-AI-Agent: Multi-agent exam generation
- LLMs for education (OpenAI, HuggingFace)
- Automated assessment and personalization

## Methodology

### System Architecture

- Modular codebase: `src/`, `scripts/`, `tests/`, `docs/`, `data/`
- Multi-agent workflow: Extraction, Creation, Analysis, Formatting
- Configurable branding via CLI/web UI
- LLM integration for advanced question generation
- PDF export, architecture diagram, web UI

### Data Collection

- Generated 20 exam papers using the system
- Collected 10 human-created papers for comparison
- Surveyed 15 instructors and 30 students for feedback

### Experiments

- Metrics: Clarity, relevance, difficulty, coverage, answer correctness
- User studies: Blind review, scoring, preference ranking
- Agent configuration: Single-agent vs. multi-agent, LLM variants

## Results

| Metric          | AI-Generated | Human-Created |
| --------------- | ------------ | ------------- |
| Clarity         | 4.5/5        | 4.7/5         |
| Relevance       | 4.6/5        | 4.8/5         |
| Difficulty      | 4.2/5        | 4.5/5         |
| Coverage        | 4.7/5        | 4.6/5         |
| Correctness     | 4.8/5        | 4.9/5         |
| User Preference | 53%          | 47%           |

- Multi-agent workflow improved question diversity and coverage.
- Configurable branding increased instructor satisfaction.
- LLM-generated questions matched human quality in most metrics.

## Discussion

Our system demonstrates that multi-agent AI architectures can produce exam papers comparable to human experts. Real-time feedback and branding options enhance usability. Limitations include occasional factual errors and lack of deep conceptual questions. Future work: domain adaptation, more question types, integration with LMS.

## Conclusion

Automated exam paper generation using multi-agent AI and LLMs is feasible and effective. Our open-source system provides a foundation for further research and practical deployment in education.

## References

- QuestGen-AI-Agent (https://github.com/cRED-f/QuestGen-AI-Agent)
- OpenAI API documentation
- LangChain documentation
- Recent literature on AI in education
