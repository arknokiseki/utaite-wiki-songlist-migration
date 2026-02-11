
import unittest
from unittest.mock import MagicMock, patch
import json
from src.parsers.llm_parser import parse_entry_with_llm

class TestLLMParser(unittest.TestCase):

    def setUp(self):
        self.raw_line = '# "Song" {{yt|abc}} (2020.01.01)'
        self.source_page = "Artist/Songs"
        self.root_artist = "Artist"
        self.sort_index = 1
        
        # Sample expected output from LLM
        self.llm_output = {
            "title": "Song",
            "youtube_id": "abc",
            "upload_date": "2020-01-01",
            "featured_artists": [],
            "version": None,
            "status": "",
            "is_original": False
        }

    @patch('src.parsers.llm_parser.CLIENT')
    def test_success_parsing(self, mock_client):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(self.llm_output)
        mock_client.chat.completions.create.return_value = mock_response

        # Run parser
        result = parse_entry_with_llm(
            self.raw_line, self.source_page, self.root_artist, self.sort_index
        )

        # Verify
        self.assertEqual(result['confidence'], 'high')
        self.assertEqual(result['parse_method'], 'llm')
        self.assertEqual(result['title'], "Song")
        self.assertEqual(result['youtube_id'], "abc")
        self.assertEqual(result['upload_date'], "2020-01-01")

        # Check call arguments
        args, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs['model'], 'gpt-4o-mini')
        messages = kwargs['messages']
        self.assertEqual(messages[1]['content'], self.raw_line)

    @patch('src.parsers.llm_parser.CLIENT')
    def test_llm_failure(self, mock_client):
        # Simulate API error
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = parse_entry_with_llm(
            self.raw_line, self.source_page, self.root_artist, self.sort_index
        )

        self.assertEqual(result['confidence'], 'low')
        self.assertTrue(result['parse_method'].startswith('llm_failed'))

    @patch('src.parsers.llm_parser.CLIENT', None)
    def test_no_client(self):
        # Simulate missing key/library
        result = parse_entry_with_llm(
            self.raw_line, self.source_page, self.root_artist, self.sort_index
        )
        
        self.assertEqual(result['confidence'], 'low')
        self.assertEqual(result['parse_method'], 'llm_failed_no_key')

if __name__ == '__main__':
    unittest.main()
