import unittest
from src.generate_cv_paper import generate_paper

class TestCVPaperGenerator(unittest.TestCase):
    def test_generate_paper_format(self):
        paper = generate_paper(num_objective=2, num_subjective=1)
        self.assertIn("Indian Institute of Science Education and Research, Bhopal", paper)
        self.assertIn("Section I: Objective Type Questions", paper)
        self.assertIn("Section II: Subjective Type Questions", paper)
        self.assertTrue(paper.count("Q1.") == 2)  # Q1 in both sections

if __name__ == "__main__":
    unittest.main()
