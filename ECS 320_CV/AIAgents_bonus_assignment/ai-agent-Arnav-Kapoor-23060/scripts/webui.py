
import gradio as gr
import random
import sys
import os
# Ensure src/ is in the Python path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from generate_cv_paper import generate_paper

def generate_exam():
    return generate_paper()

def show_solutions():
    with open("ECS320_CV_Endsem_Paper_Solutions.txt") as f:
        return f.read()

def show_syllabus():
    with open("ECS320_CV_Syllabus.txt") as f:
        return f.read()

with gr.Blocks() as demo:
    gr.Markdown("# Computer Vision Exam Paper Generator (ECS 320)")
    with gr.Tab("Generate Paper"):
        paper_output = gr.Textbox(label="Exam Paper", lines=30)
        gr.Button("Generate").click(lambda: generate_exam(), outputs=paper_output)
    with gr.Tab("Solutions"):
        solutions_output = gr.Textbox(label="Solutions", lines=20)
        gr.Button("Show Solutions").click(lambda: show_solutions(), outputs=solutions_output)
    with gr.Tab("Syllabus"):
        syllabus_output = gr.Textbox(label="Syllabus", lines=20)
        gr.Button("Show Syllabus").click(lambda: show_syllabus(), outputs=syllabus_output)

demo.launch(share=True)
