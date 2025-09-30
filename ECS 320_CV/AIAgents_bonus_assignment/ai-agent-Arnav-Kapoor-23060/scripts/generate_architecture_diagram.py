from graphviz import Digraph

diagram = Digraph('AI_Agent_CV_Question_Paper', format='pdf')

diagram.attr(rankdir='LR', size='8,5')

diagram.node('KB', 'Knowledge Base / Curriculum Mapping')
diagram.node('QG', 'Question Generator (NLP/LLM)')
diagram.node('DC', 'Difficulty & Quality Control')
diagram.node('PS', 'Paper Structuring Module')
diagram.node('UI', 'User Interface (Instructor/Student)')
diagram.node('EV', 'Evaluation & Feedback Loop')
diagram.node('SEC', 'Security & Academic Integrity')

diagram.edges([
    ('KB', 'QG'),
    ('QG', 'DC'),
    ('DC', 'PS'),
    ('PS', 'UI'),
    ('UI', 'EV'),
    ('EV', 'KB'), # Feedback loop
    ('PS', 'SEC'),
    ('UI', 'SEC')
])

diagram.attr(label='AI Agent for Computer Vision Question Paper Generation', labelloc='t', fontsize='20')

diagram.render('AI_Agent_CV_Question_Paper_Architecture', view=False)
print("System architecture diagram generated: AI_Agent_CV_Question_Paper_Architecture.pdf")
