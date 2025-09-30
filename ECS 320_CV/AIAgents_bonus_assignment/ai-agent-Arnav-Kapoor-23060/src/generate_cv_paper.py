
import random
import argparse

# Optional: LLM integration for advanced question generation
try:
    import openai
except ImportError:
    openai = None


from typing import Dict, List, Any

def generate_llm_question(prompt: str, api_key: str = None) -> str:
    """Generate a question using OpenAI's GPT-3/4 API."""
    if not openai:
        return "[OpenAI API not installed. Install with 'pip install openai']"
    if not api_key:
        return "[No OpenAI API key provided]"
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI API error: {e}]"

OBJECTIVE_QUESTIONS = [
    {
        "question": "Which layer in a CNN is primarily responsible for feature extraction?",
        "options": ["Convolutional", "Pooling", "Fully Connected", "Dropout"],
        "answer": "Convolutional"
    },
    {
        "question": "What is the main purpose of Non-Maximum Suppression in object detection?",
        "options": ["Reduce false positives", "Merge overlapping boxes", "Increase recall", "Normalize images"],
        "answer": "Merge overlapping boxes"
    },
    {
        "question": "Which loss function is commonly used for image segmentation tasks?",
        "options": ["Cross-Entropy", "Mean Squared Error", "Hinge Loss", "Triplet Loss"],
        "answer": "Cross-Entropy"
    },
    {
        "question": "Which architecture is best suited for sequential image data (video)?",
        "options": ["RNN", "CNN", "LSTM", "GAN"],
        "answer": "LSTM"
    },
    {
        "question": "Which technique is used for unsupervised feature learning in CV?",
        "options": ["Autoencoders", "SVM", "Random Forest", "KNN"],
        "answer": "Autoencoders"
    }
]

SUBJECTIVE_QUESTIONS = [
    "Discuss the challenges and solutions in domain adaptation for computer vision tasks.",
    "Explain the architecture and applications of Vision Transformers in detail.",
    "Describe the process and significance of 3D reconstruction from multiple images.",
    "How does explainable AI impact the deployment of computer vision models in critical applications?",
    "Compare and contrast zero-shot and few-shot learning approaches in computer vision.",
    "Design a pipeline for medical image segmentation using deep learning. Discuss the evaluation metrics.",
    "Explain the role of self-supervised learning in advancing computer vision research.",
    "Describe the steps involved in building a visual SLAM system for autonomous navigation.",
    "Discuss the integration of multi-modal data (vision and language) for scene understanding.",
    "Evaluate the strengths and limitations of GANs in image synthesis and augmentation."
]

def extract_requirements(code: str = "ECS 320", exam_name: str = "MidSemester Exam", date: str = "August 17, 2025", professor: str = "Dr. Akshay Aggarwal", max_marks: int = 30, institute: str = "Indian Institute of Science Education and Research, Bhopal") -> Dict[str, Any]:
    return {
        "institute": institute,
        "course_name": f"{code} Computer Vision",
        "course_code": code,
        "exam_name": exam_name,
        "date": date,
        "max_marks": max_marks,
        "professor": professor
    }

def create_questions(num_objective: int = 5, num_subjective: int = 2) -> Dict[str, Any]:
    obj_qs = random.sample(OBJECTIVE_QUESTIONS, num_objective)
    subj_qs = random.sample(SUBJECTIVE_QUESTIONS, num_subjective)
    return {
        "objective": obj_qs,
        "subjective": subj_qs
    }

def analyze_questions(questions: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder for analysis logic (could be expanded with LLM or rules)
    # For now, just return the questions as-is
    return questions

def format_paper(requirements: Dict[str, Any], questions: Dict[str, Any]) -> str:
    paper = (
        "------------------------------------------------------------\n"
        f"        {requirements['institute']}\n"
        "------------------------------------------------------------\n"
        "\n"
        f"Course Name: {requirements['course_name']}\n"
        f"Course Code: {requirements['course_code']}\n"
        f"Exam Name: {requirements['exam_name']}\n"
        f"Date of Exam: {requirements['date']}\n"
        f"Maximum Marks: {requirements['max_marks']}\n"
        f"Professor: {requirements['professor']}\n"
        "\n"
        "------------------------------------------------------------\n"
    )
    paper += "\nSection I: Objective Type Questions (MCQs) [10 marks]\n\n"
    option_labels = ['A', 'B', 'C', 'D']
    for i, q in enumerate(questions['objective'], 1):
        paper += f"Q{i}. {q['question']} (2 marks)\n"
        for label, opt in zip(option_labels, q['options']):
            paper += f"   {label}. {opt}\n"
        paper += "\n"
    paper += "------------------------------------------------------------\n\n"
    paper += "Section II: Subjective Type Questions [20 marks]\n\n"
    for i, q in enumerate(questions['subjective'], 1):
        paper += f"Q{i}. {q} (10 marks)\n\n"
    paper += "------------------------------------------------------------\n"
    return paper

def generate_paper(code: str = "ECS 320", num_objective: int = 5, num_subjective: int = 2, exam_name: str = "MidSemester Exam", date: str = "August 17, 2025", professor: str = "Dr. Akshay Aggarwal", max_marks: int = 30, institute: str = "Indian Institute of Science Education and Research, Bhopal") -> str:
    requirements = extract_requirements(code, exam_name, date, professor, max_marks, institute)
    questions = create_questions(num_objective, num_subjective)
    analyzed = analyze_questions(questions)
    return format_paper(requirements, analyzed)

def main():
    parser = argparse.ArgumentParser(description="Generate Computer Vision exam paper, solutions, syllabus, and LLM-based questions.")
    parser.add_argument('--objective', type=int, default=5, help='Number of objective questions')
    parser.add_argument('--subjective', type=int, default=2, help='Number of subjective questions')
    parser.add_argument('--paper', action='store_true', help='Generate exam paper')
    parser.add_argument('--solutions', action='store_true', help='Generate solutions file')
    parser.add_argument('--syllabus', action='store_true', help='Generate syllabus file')
    parser.add_argument('--all', action='store_true', help='Generate all files')
    parser.add_argument('--llm', type=str, help='Prompt for LLM-based question generation')
    parser.add_argument('--api_key', type=str, help='OpenAI API key for LLM integration')
    parser.add_argument('--institute', type=str, default="Indian Institute of Science Education and Research, Bhopal", help='Institute name for branding')
    parser.add_argument('--professor', type=str, default="Dr. Akshay Aggarwal", help='Professor name')
    parser.add_argument('--exam_name', type=str, default="MidSemester Exam", help='Exam name')
    parser.add_argument('--date', type=str, default="August 17, 2025", help='Date of exam')
    parser.add_argument('--max_marks', type=int, default=30, help='Maximum marks')
    args = parser.parse_args()

    if args.llm:
        print("LLM-generated question:")
        print(generate_llm_question(args.llm, api_key=args.api_key))

    if args.paper or args.all:
        paper = generate_paper(
            num_objective=args.objective,
            num_subjective=args.subjective,
            exam_name=args.exam_name,
            date=args.date,
            professor=args.professor,
            max_marks=args.max_marks,
            institute=args.institute
        )
        with open("ECS320_CV_Endsem_Paper.txt", "w") as f:
            f.write(paper)
        print("Paper generated: ECS320_CV_Endsem_Paper.txt")

    if args.solutions or args.all:
        solutions = (
            f"------------------------------------------------------------\n"
            f"        {args.institute}\n"
            "------------------------------------------------------------\n"
            "\n"
            f"Course Name: ECS 320 Computer Vision\n"
            f"Course Code: ECS 320\n"
            f"Exam Name: {args.exam_name} Solutions\n"
            f"Date of Exam: {args.date}\n"
            f"Professor: {args.professor}\n"
            "\n"
            "------------------------------------------------------------\n"
        )
        solutions += "\nSection I: Objective Type Questions Solutions\n"
        solutions += "Q1: A. Cross-Entropy\nQ2: A. Convolutional\nQ3: C. LSTM\nQ4: B. Merge overlapping boxes\nQ5: A. Autoencoders\n\n"
        solutions += "Section II: Subjective Type Questions Solutions\n"
        solutions += "Q1: Zero-shot learning enables models to recognize objects/classes not seen during training by leveraging semantic relationships, while few-shot learning focuses on learning from a very small number of examples. Both are crucial for scalable computer vision, but zero-shot relies more on external knowledge (e.g., attributes, word vectors), whereas few-shot uses meta-learning or transfer learning techniques.\n\n"
        solutions += "Q2: 3D reconstruction from multiple images involves extracting depth and structure by matching features across images, triangulating points, and building a 3D model. It is significant for applications like AR/VR, robotics, and medical imaging, as it allows for spatial understanding and interaction with real-world environments.\n"
        with open("ECS320_CV_Endsem_Paper_Solutions.txt", "w") as f:
            f.write(solutions)
        print("Solutions generated: ECS320_CV_Endsem_Paper_Solutions.txt")

    if args.syllabus or args.all:
        syllabus = (
            f"------------------------------------------------------------\n"
            f"        {args.institute}\n"
            "------------------------------------------------------------\n"
            "\n"
            f"Course Name: ECS 320 Computer Vision\n"
            f"Course Code: ECS 320\n"
            f"Exam Name: Syllabus\n"
            f"Date of Exam: {args.date}\n"
            f"Professor: {args.professor}\n"
            "\n"
            "------------------------------------------------------------\n"
        )
        syllabus += "\nSyllabus for ECS 320 Computer Vision:\n"
        syllabus += "- Fundamentals of Computer Vision\n- Image Processing and Feature Extraction\n- Convolutional Neural Networks (CNNs)\n- Object Detection and Recognition\n- Image Segmentation\n- Transfer Learning in CV\n- Generative Adversarial Networks (GANs)\n- 3D Vision and Reconstruction\n- Visual SLAM\n- Deep Learning for Video Analysis\n- Explainable AI in CV\n- Medical Image Analysis\n- Self-supervised Learning\n- Vision Transformers\n- Zero-shot and Few-shot Learning\n- Domain Adaptation\n- Multi-modal Learning (Vision + Language)\n"
        with open("ECS320_CV_Syllabus.txt", "w") as f:
            f.write(syllabus)
        print("Syllabus generated: ECS320_CV_Syllabus.txt")

if __name__ == "__main__":
    main()
