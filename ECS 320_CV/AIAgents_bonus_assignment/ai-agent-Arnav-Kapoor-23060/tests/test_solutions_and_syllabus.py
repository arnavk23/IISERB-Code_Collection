import unittest
from src.generate_cv_paper import generate_paper
import os

class TestCVSolutionsSyllabus(unittest.TestCase):
    def test_generate_solutions_file(self):
        # Generate solutions file
        os.system('python src/generate_cv_paper.py --solutions')
        self.assertTrue(os.path.exists('ECS320_CV_Endsem_Paper_Solutions.txt'))
        with open('ECS320_CV_Endsem_Paper_Solutions.txt') as f:
            content = f.read()
        self.assertIn('MidSemester Exam Solutions', content)
        self.assertIn('Section I: Objective Type Questions Solutions', content)

    def test_generate_syllabus_file(self):
        # Generate syllabus file
        os.system('python src/generate_cv_paper.py --syllabus')
        self.assertTrue(os.path.exists('ECS320_CV_Syllabus.txt'))
        with open('ECS320_CV_Syllabus.txt') as f:
            content = f.read()
        self.assertIn('Syllabus for ECS 320 Computer Vision', content)
        self.assertIn('Vision Transformers', content)

if __name__ == "__main__":
    unittest.main()
