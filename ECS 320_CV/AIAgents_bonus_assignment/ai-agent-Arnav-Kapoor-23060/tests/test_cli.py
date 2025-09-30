import unittest
import subprocess
import os

class TestCLI(unittest.TestCase):
    def test_generate_paper_cli(self):
        subprocess.run(['python', 'src/generate_cv_paper.py', '--paper', '--objective', '2', '--subjective', '1'])
        self.assertTrue(os.path.exists('ECS320_CV_Endsem_Paper.txt'))
        with open('ECS320_CV_Endsem_Paper.txt') as f:
            content = f.read()
        self.assertIn('Indian Institute of Science Education and Research, Bhopal', content)
        self.assertIn('Section I: Objective Type Questions', content)

    def test_generate_all_cli(self):
        subprocess.run(['python', 'src/generate_cv_paper.py', '--all'])
        self.assertTrue(os.path.exists('ECS320_CV_Endsem_Paper_Solutions.txt'))
        self.assertTrue(os.path.exists('ECS320_CV_Syllabus.txt'))

if __name__ == "__main__":
    unittest.main()
