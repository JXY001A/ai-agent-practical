from typing import Any, Dict, List, Optional, Tuple
import time

def _get_mock_search_results(query: str="Sam Altman current position OpenAI CEO 2025 ") -> Dict[str, Any]:
        """
        Get mock search results for testing without API key
        
        Args:
            query: Search query
            
        Returns:
            Mock search results
        """
        # Mock results for OpenAI co-founders
        mock_data = {
            "openai": [
                {
                    'title': 'OpenAI - Wikipedia',
                    'url': 'https://en.wikipedia.org/wiki/OpenAI',
                    'snippet': 'OpenAI was founded in 2015 by Sam Altman, Elon Musk, Ilya Sutskever, Greg Brockman, Wojciech Zaremba, and John Schulman...',
                    'content': '''OpenAI was founded in December 2015 by Sam Altman, Elon Musk, Ilya Sutskever, Greg Brockman, Wojciech Zaremba, and John Schulman.
                    
                    The organization was founded with the goal of advancing digital intelligence in a way that benefits humanity. 

                    Current Status of Co-founders (as of 2024):
                    - Sam Altman: CEO of OpenAI (returned after brief departure in November 2023)
                    - Elon Musk: Left OpenAI board in 2018, founded xAI in 2023
                    - Ilya Sutskever: Former Chief Scientist, left OpenAI in May 2024, co-founded Safe Superintelligence Inc.
                    - Greg Brockman: President and Chairman of OpenAI
                    - Wojciech Zaremba: Head of Language and Code Generation at OpenAI
                    - John Schulman: Co-founder, left OpenAI in August 2024 to join Anthropic

                    Additional early members:
                    - Andrej Karpathy: Former Director of AI at Tesla, briefly returned to OpenAI, now independent
                    - Dario Amodei: Left to co-found Anthropic in 2021
                    - Daniela Amodei: Left to co-found Anthropic in 2021'''
                }
            ],
            "sam altman": [
                {
                    'title': 'Sam Altman - CEO of OpenAI',
                    'url': 'https://example.com/sam-altman',
                    'snippet': 'Sam Altman is the CEO of OpenAI...',
                    'content': 'Sam Altman is currently the CEO of OpenAI. He briefly left the company in November 2023 but returned after employee protests. He is also known for his work at Y Combinator and various investments in startups.'
                }
            ],
            "elon musk": [
                {
                    'title': 'Elon Musk launches xAI',
                    'url': 'https://example.com/elon-musk-ai',
                    'snippet': 'Elon Musk founded xAI in 2023...',
                    'content': 'Elon Musk, who co-founded OpenAI in 2015, left the board in 2018 citing conflicts of interest with Tesla\'s AI development. In 2023, he founded xAI, a new AI company focused on understanding the universe. He is also CEO of Tesla, SpaceX, and owner of X (formerly Twitter).'
                }
            ],
            "ilya sutskever": [
                {
                    'title': 'Ilya Sutskever launches Safe Superintelligence',
                    'url': 'https://example.com/ilya-sutskever',
                    'snippet': 'Ilya Sutskever left OpenAI to start SSI...',
                    'content': 'Ilya Sutskever, former Chief Scientist at OpenAI, left the company in May 2024 after nearly a decade. He co-founded Safe Superintelligence Inc. (SSI) with Daniel Gross and Daniel Levy, focusing on building safe AGI.'
                }
            ]
        }
        query_lower = query.lower()
        for key in mock_data:
            if key in query_lower:
                results = []
                for item in mock_data[key]:
                    results.append({
                        'title': item['title'],
                        'url': item['url'],
                        'snippet': item['snippet'],
                        'content': item['content'],
                        'content_length': len(item['content']),
                        'fetch_success': True
                    })
                
                return {
                    'query': query,
                    'num_results': len(results),
                    'results': results,
                    'timestamp': time.time(),
                    'mock': True
                }
