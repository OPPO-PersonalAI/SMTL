#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 OPPO. PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Any, Optional, Tuple
import os
import sys
import requests
import json
import time
from json_repair import json_repair
from transformers import AutoTokenizer

# Handle imports when running as script vs as module
# Add parent directory to path if running as script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    # Try relative import first (when imported as module)
    from .tools import Tool
    from .models import OpenAIServerModel
except ImportError:
    # Fall back to absolute import (when run as script)
    from FlashOAgents.tools import Tool
    from FlashOAgents.models import OpenAIServerModel

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

def read_page(url: str, max_retries: int = 3) -> str:
    """Read and return the content of a webpage using Jina reader."""
    jina_url = f'https://r.jina.ai/{url}'
    headers = {
        'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}',
        'X-Engine': 'browser',
        'X-Return-Format': 'markdown',
        "X-Remove-Selector": "header, .class, #id",
        "X-Retain-Images": "none",
        'X-Timeout': '120',
        'X-Token-Budget': '200000',
    }

    # Use a session for connection pooling and better SSL handling
    session = requests.Session()
    # Configure SSL to be more lenient with connection issues
    session.verify = True  # Keep SSL verification enabled for security
    
    attempt = 0
    consecutive_ssl_errors = 0
    try:
        while attempt < max_retries:
            try:
                # Use shorter timeout for connection, but allow longer for read
                # This helps avoid hanging on slow connections
                response = session.get(
                    jina_url,
                    headers=headers,
                    timeout=(20, 120),  # (connect timeout, read timeout)
                    stream=False  # Don't stream to avoid SSL issues with streaming
                )
                if response.ok:
                    return response.text
                else:
                    error_msg = response.text
                    # Non-2xx status codes are considered server/request failures and are counted towards retries
                    if attempt == max_retries - 1:
                        return f"Error: {error_msg}"
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    attempt += 1
                    continue
            except requests.exceptions.SSLError as ssl_error:
                # SSL errors are related to local network/VPN issues and are not counted towards retry attempts
                consecutive_ssl_errors += 1
                # Wait longer with exponential backoff
                wait_time = 2 ** min(consecutive_ssl_errors, 5)
                time.sleep(wait_time)
                # Do not increment the attempt count; continue retrying
                continue
            except requests.exceptions.Timeout as timeout_error:
                # Timeout errors: Counted towards retry attempts
                if attempt == max_retries - 1:
                    return f"Error: Request timed out after {max_retries} attempts: {str(timeout_error)}"
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                attempt += 1
            except requests.RequestException as e:
                # Other request exceptions: counted towards retry attempts
                if attempt == max_retries - 1:
                    return f"Error: Reading page after {max_retries} attempts: {str(e)}"
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                attempt += 1
    finally:
        session.close()
    
    return "Error: Unexpected error in page reading"

def web_search_google_serper(
    query: str, 
    filter_year: Optional[int] = None, 
    serp_num: int = 3, 
    max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], str]:
    """Perform web search using Google Serper API."""
    def _contains_chinese_basic(query: str) -> bool:
        return any('\u4E00' <= char <= '\u9FFF' for char in query)
    
    if not query.strip():
        return [], "Query is empty. Please provide a valid search query."

    if _contains_chinese_basic(query):
        payload = json.dumps({
            "q": query,
            "location": "China",
            "gl": "cn",
            "hl": "zh-cn",
            "num": serp_num
        })
    else:
        payload = json.dumps({
            "q": query,
            "location": "United States",
            "gl": "us",
            "hl": "en",
            "num": serp_num
        })

    url = "https://google.serper.dev/search"
    headers = {
        'X-API-KEY': os.getenv("WEB_SEARCH_SERPER_API_KEY"),
        'Content-Type': 'application/json'
    }

    attempt = 0
    consecutive_ssl_errors = 0
    while attempt < max_retries:
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=120)
            if not response.ok:
                error_msg = response.text
                if attempt == max_retries - 1:
                    return [], f"{error_msg}"
                time.sleep(1)
                attempt += 1
                continue
            results = response.json()

            if "organic" not in results or not results["organic"]:
                year_filter_msg = f" with year filter={filter_year}" if filter_year else ""
                return [], f"No results found for '{query}'{year_filter_msg}. The search query may be too strict. Consider: (1) Strategic Major Adjustment - re-read the original query conditions, analyze what information you've already found, exclude ineffective approaches, and find new search directions; (2) Search Query Minor Adjustment - relax strict constraints (e.g., remove year/site restrictions, use alternative search terms or synonyms)."
            
            search_results = []
            for idx, page in enumerate(results["organic"], 1):
                search_results.append({
                    "idx": idx,
                    "title": page.get("title", "No title"),
                    "date": f"Date published: {page['date']}" if "date" in page else "",
                    "snippet": f"{page.get('snippet', 'No snippet')}",
                    "source": f"Source: {page.get('source', 'Unknown source')}",
                    "link": page.get('link', '#')
                })
            
            return search_results, ""
        except requests.exceptions.SSLError as ssl_error:
            # SSL errors do not count towards retry attempts; exponential backoff
            consecutive_ssl_errors += 1
            wait_time = 2 ** min(consecutive_ssl_errors, 5)
            time.sleep(wait_time)
            continue
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                return [], f"Search failed after {max_retries} attempts: {str(e)}"
            time.sleep(1)
            attempt += 1
    
    return [], "Unexpected error in web search"

class WebSearchTool(Tool):
    name = "web_search"
    description = "Use for BROAD exploration to discover potential information sources. Perform a web search query and return search results (URLs and snippets)."
    inputs = {
        "query": {
            "type": "string", 
            "description": "A single web search query. CRITICAL: Should be SHORT keyword phrases (NOT full sentences or questions) that combine multiple critical clues for HIGH-HIT potential. Phrases are split by space. If needed, phrases with quotas are available and they target at the websites that should contain these phrases, otherwise, do not use quotas if you don't want to emphasize the phrases. Good example: 'artist trio 1984 album bassist vocalist' (combines multiple clues). Bad example: 'What artist was in a trio that released an album in 1984?' (full question). Purpose: Cast a wide net to find relevant URLs and information snippets."
        }
    }
    output_type = "string"

    def __init__(self, serp_num=10):
        super().__init__()
        self.tool_name = "web_search"
        self.serp_num = serp_num

    def forward(self, query: str) -> str:
        """Execute a single web search and return formatted results."""
        query = query.strip()
        
        if not query:
            return "Error: No valid query provided."
        
        # query_cleaned = query.replace('"', "").replace("'", "")
        search_results, error_msg = web_search_google_serper(query, serp_num=self.serp_num)

        # Add query header
        query_header = f"{'='*60}\nQuery: {query}\n{'='*60}"
        
        if error_msg:
            # Check if it's a "no results" type error that needs strategy adjustment guidance
            if "No results found" in error_msg or "no results" in error_msg.lower():
                return f"{query_header}\nError: {error_msg}\n\nStrategy Adjustment Needed: (1) Strategic Major Adjustment - re-read query conditions, analyze existing information, find new search directions; (2) Search Query Minor Adjustment - relax strict constraints (site/year restrictions, quoted phrases) or use alternative search terms."
            return f"{query_header}\nError: {error_msg}"
        
        if not search_results:
            return f"{query_header}\nError: No search results found. The search query may be too strict or the search direction may be ineffective. Consider: (1) Strategic Major Adjustment - re-read the original query conditions, analyze existing information, exclude failed approaches, and identify new search directions; (2) Search Query Minor Adjustment - if the query includes strict constraints (site/year restrictions, quoted phrases), relax these constraints or use alternative search terms."
        
        # Format results
        formatted_results = []
        for result in search_results:
            formatted_results.append(
                f"{result['idx']}. [{result['title']}]({result['link']})"
                f"{result['date']}{result['source']}\n"
                f"   {result['snippet'].strip()}"
            )
        
        # Add reminder about parallel crawl_page verification when multiple results are found
        result_text = f"{query_header}"
        # If we have results and error_msg is a notice (fuzzy-match), add it before results
        if error_msg and search_results:
            result_text += f"\nNote: {error_msg}\n"
        else:
            result_text += "\n"
        result_text += "\n\n".join(formatted_results)
        if len(search_results) > 1:
            result_text += f"\n\n[Note: Multiple URLs found. Carefully analyze each URL's title, snippet, and source to determine which ones are potentially relevant to your query requirements and reasoning needs. For URLs that show promise (match query conditions or align with your reasoning), use crawl_page to verify them IN PARALLEL. Do not miss any URLs with potential, but also be selective - only crawl URLs that genuinely appear relevant.]"
        elif len(search_results) == 1:
            result_text += f"\n\n[Note: A URL found. Analyze whether this URL's title and snippet suggest it may contain information relevant to your query requirements. If it shows potential, use crawl_page to extract detailed information from this source.]"
        
        return result_text

class CrawlPageTool(Tool):
    name = "crawl_page"
    description = "Use for DEEP exploration of specific promising sources found via web_search. Access a webpage using the provided URL and extract detailed information relevant to a specific goal."
    inputs = {
        "url": {
            "type": "string",
            "description": "A single URL of a specific webpage to visit. This should be a promising URL discovered via web_search that likely contains detailed information. Example: 'https://en.wikipedia.org/wiki/Artist_Name' to extract information from an authoritative source."
        },
        "query": {
            "type": "string",
            "description": "The precise information extraction goal describing what specific data you need to extract from the webpage. This should be a complete, clear sentence with specific details. Good example: 'Extract artist's real name, birth date, debut album release year, and album tracklist with non-English titles'. Bad example: 'Find information about the artist' (too vague)."
        }
    }
    output_type = "string"
    
    # Class-level tokenizer cache
    _tokenizer = None
    _tokenizer_path = None
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__()
        self.tool_name = "crawl_page"
        self.model = model
    
    @classmethod
    def _get_tokenizer(cls):
        """Load and cache tokenizer from config directory."""
        
        # Primary path: data_workflow/config (relative to this file)
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_file_dir)
        config_path = os.path.join(project_root, "config")
        # Fallback path: tokenizer/qwen3/Qwen3-4B (relative to current working directory)
        # Assumes the process is launched from SMTL-main root.
        fallback_rel_path = os.path.join("tokenizer", "qwen3/Qwen3-4B")
        fallback_config_path = os.path.abspath(fallback_rel_path)
        
        # Check if tokenizer path changed or not loaded
        if cls._tokenizer is None or cls._tokenizer_path != config_path:
            try:
                if os.path.exists(config_path) and os.path.exists(os.path.join(config_path, "tokenizer.json")):
                    cls._tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)
                    cls._tokenizer_path = config_path
                else:
                    # Fallback: try a tokenizer directory under current working directory
                    if os.path.exists(fallback_config_path) and os.path.exists(os.path.join(fallback_config_path, "tokenizer.json")):
                        cls._tokenizer = AutoTokenizer.from_pretrained(fallback_config_path, trust_remote_code=True)
                        cls._tokenizer_path = fallback_config_path
            except Exception as e:
                print(f"Warning: Failed to load tokenizer from {config_path}: {e}")
                cls._tokenizer = None
                cls._tokenizer_path = None
        
        return cls._tokenizer

    @staticmethod
    def truncate_text(text: str, max_length: int = None, max_tokens: int = 200000) -> str:
        """
        Truncate text based on token count using tokenizer.
        
        Args:
            text: The text to truncate
            max_length: Deprecated parameter, kept for backward compatibility
            max_tokens: Maximum number of tokens (default: 250000)
        
        Returns:
            Truncated text if it exceeds max_tokens, otherwise original text
        """
        if not text:
            return text
        
        tokenizer = CrawlPageTool._get_tokenizer()
        
        # If tokenizer is not available, fall back to character-based truncation
        if tokenizer is None:
            if max_length is not None:
                return text if len(text) <= max_length else text[:max_length] + "...(truncated)"
            # If no tokenizer and no max_length, return as-is but warn
            print("Warning: Tokenizer not available, cannot perform token-based truncation")
            return text
        
        # Count tokens
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_count = len(tokens)
            
            # If within limit, return original text
            if token_count <= max_tokens:
                return text
            
            # Truncate tokens directly and decode back to text
            # This is more efficient: only encode once, decode once
            truncated_tokens = tokens[:max_tokens]
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            # Verify the decoded text doesn't exceed token limit (decode might add padding/special tokens)
            verify_tokens = tokenizer.encode(truncated_text, add_special_tokens=False)
            if len(verify_tokens) > max_tokens:
                # If it still exceeds, truncate more aggressively
                truncated_tokens = verify_tokens[:max_tokens]
                truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            # Try to find a good breaking point in the original text to avoid cutting mid-word
            # Find where the truncated_text roughly aligns in the original text
            truncated_len = len(truncated_text)
            if truncated_len < len(text):
                # Search for a good break point near the end of truncated text
                search_start = max(0, int(truncated_len * 0.85))  # Search in last 15%
                search_end = min(len(text), truncated_len + 100)  # Allow some lookahead
                
                # Try to find natural break points (paragraph, sentence, word boundaries)
                for break_char in ['\n\n', '\n', '. ', '。', '! ', '? ', ' ', '']:
                    last_break = text.rfind(break_char, search_start, search_end)
                    if last_break >= search_start:
                        # Verify this position doesn't exceed token limit
                        candidate_text = text[:last_break + len(break_char)]
                        candidate_tokens = tokenizer.encode(candidate_text, add_special_tokens=False)
                        if len(candidate_tokens) <= max_tokens:
                            truncated_text = candidate_text
                            break
            
            return truncated_text + "...(truncated)"
            
        except Exception as e:
            print(f"Warning: Error during tokenization: {e}, falling back to character-based truncation")
            if max_length is not None:
                return text if len(text) <= max_length else text[:max_length] + "...(truncated)"
            return text

    def get_summary_prompt_single(self, query: str, url: str, content: str) -> str:
        """Generate prompt for single webpage content summarization."""
        return f"""Please process the following webpage content and extract information relevant to the goal:

## **Webpage URL**
{url}

## **Webpage Content** 
{content}

## **Information Goal**
{query}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content. Never miss any important information. Output the **full original context** of the content as much as possible (can be more than three paragraphs).
3. **Summary Output**: Organize findings into a concise but comprehensive summary with logical flow, prioritizing clarity and relevance to the goal.

**Final Output Format using JSON with "rationale", "evidence", "summary" fields**
""".strip()

    def get_summary_prompt_multi(self, query: str, url_contents: List[Dict[str, str]]) -> str:
        """Generate prompt for multiple webpages content synthesis."""
        content_sections = []
        for idx, item in enumerate(url_contents, 1):
            content_sections.append(f"""### Source {idx}: {item['url']}
{item['content']}""")
        
        all_content = "\n\n".join(content_sections)
        
        return f"""Please process content from multiple webpages and synthesize information relevant to the user's goal:

## **Multiple Webpage Contents**
{all_content}

## **Information Goal**
{query}

## **Task Guidelines**
1. **Cross-Source Analysis**: Review content from all {len(url_contents)} sources and identify information relevant to the goal
2. **Information Synthesis**: 
   - Extract key facts, data, and insights from each source
   - Note agreements and discrepancies across sources
   - Identify unique contributions from each source
   - Never miss important information from any source
3. **Comprehensive Summary**: 
   - Synthesize findings into a coherent summary addressing the goal
   - Cite source numbers (e.g., "According to Source 1...", "Source 2 confirms...")
   - Highlight corroborated information vs. unique claims
   - Note any contradictions or inconsistencies found

**Final Output Format using JSON with these fields:**
- "rationale": Your reasoning for selecting relevant information from each source
- "evidence": Detailed evidence extracted from each source (organized by source)
- "summary": Comprehensive synthesis of all sources addressing the information goal
""".strip()

    def retry_predict(self, prompt: str, max_retries: int = 3) -> str:
        """Retry model prediction with exponential backoff."""
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(max_retries):
            try:
                response = self.model(messages)
                if hasattr(response, 'content'):
                    content = response.content
                    return content.strip() if isinstance(content, str) else str(content)
                return str(response)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error: Content extraction failed: {str(e)}"
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        
        return "Error: Content extraction failed after multiple attempts"

    def forward(self, url: str, query: str) -> str:
        """Crawl a single webpage and extract relevant content."""
        url = url.strip()
        
        if not url:
            return (
                "Error: No valid URL provided.\n"
                "Suggestion: Please provide a full URL starting with http:// or https:// (e.g., copy the exact page link) and retry."
            )
        
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return (
                "Error: Invalid URL format. Must start with http:// or https://\n"
                "Suggestion: Add the correct scheme (http:// or https://), verify the URL is reachable in a browser, and retry."
            )

        # Get page content
        page_content = read_page(url)
        if page_content.startswith("Error"):
            return (
                f"Failed to crawl {url}: {page_content}\n"
                "Suggestion: The page may block crawlers or be temporarily unavailable. Try a different URL/mirror, reduce redirects, or retry later. "
                "If the page is dynamic (JS-heavy), try a more static source (e.g., Wikipedia, PDF, or an archived copy)."
            )
        
        # Process single page - allocate all 200000 tokens
        truncated_content = self.truncate_text(page_content, max_length=None, max_tokens=200000)
        prompt = self.get_summary_prompt_single(query, url, truncated_content)
        
        summary = self.retry_predict(prompt)
        if isinstance(summary, str) and summary.startswith("Error"):
            return (
                f"{summary}\n"
                "Suggestion: Try simplifying the goal (be more specific/short), retry later, or use a different page. "
                "If the content is very long/noisy, point to a more focused URL (e.g., a section anchor, a PDF, or a shorter article)."
            )
        return summary
