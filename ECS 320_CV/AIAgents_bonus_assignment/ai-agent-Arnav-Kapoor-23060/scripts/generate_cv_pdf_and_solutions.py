from fpdf import FPDF

# Read the paper from the text file
with open("ECS320_CV_Endsem_Paper.txt", "r") as f:
    paper_text = f.read()

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Indian Institute of Science Education and Research, Bhopal', 0, 1, 'C')
        self.ln(2)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', '', 12)
for line in paper_text.split('\n'):
    pdf.multi_cell(0, 10, line)

pdf.output("ECS320_CV_Endsem_Paper.pdf")
print("PDF generated: ECS320_CV_Endsem_Paper.pdf")

# Solutions for the generated paper
solutions = """
Section I: Objective Type Questions Solutions
Q1: A. Cross-Entropy
Q2: A. Convolutional
Q3: C. LSTM
Q4: B. Merge overlapping boxes
Q5: A. Autoencoders

Section II: Subjective Type Questions Solutions
Q1: Zero-shot learning enables models to recognize objects/classes not seen during training by leveraging semantic relationships, while few-shot learning focuses on learning from a very small number of examples. Both are crucial for scalable computer vision, but zero-shot relies more on external knowledge (e.g., attributes, word vectors), whereas few-shot uses meta-learning or transfer learning techniques.

Q2: 3D reconstruction from multiple images involves extracting depth and structure by matching features across images, triangulating points, and building a 3D model. It is significant for applications like AR/VR, robotics, and medical imaging, as it allows for spatial understanding and interaction with real-world environments.
"""
with open("ECS320_CV_Endsem_Paper_Solutions.txt", "w") as f:
    f.write(solutions)
print("Solutions generated: ECS320_CV_Endsem_Paper_Solutions.txt")
