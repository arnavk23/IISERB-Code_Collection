import unittest
from src.generate_cv_paper import generate_llm_question

class TestLLMIntegration(unittest.TestCase):
    def test_llm_no_api(self):
        # Should return error message if no API key or openai not installed
        result = generate_llm_question("Generate a computer vision MCQ.")
        self.assertTrue(
            "No OpenAI API key provided" in result or
            "OpenAI API not installed" in result
        )

    def test_llm_no_openai(self):
        # Should return error message if openai not installed
        import sys
        sys.modules['openai'] = None
        result = generate_llm_question("Generate a computer vision MCQ.", api_key="fake")
        self.assertIn("OpenAI API not installed", result)

if __name__ == "__main__":
    unittest.main()
