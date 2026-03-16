#!/usr/bin/env python
# coding=utf-8
# Copyright 2026 OPPO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import atexit
import json
import logging
import os
import time
from typing import Dict, Optional, Any, List
import random
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from dotenv import load_dotenv
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SERPER_TIMEOUT = 60
CRAWL_PAGE_TIMEOUT = 500
MAX_CONCURRENT_PAGES = 5  # New: Control maximum concurrent pagination requests to avoid rate limiting

import aiosqlite  # Async SQLite
import hashlib
from datetime import datetime

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class AsyncWebSearchCache:
    """High-performance LRU cache based on async SQLite for caching web search queries and their results"""

    def __init__(self, db_path: str = "web_search_cache.db", max_size: int = 100000, ttl: int = 8640000):
        """
        Initialize search cache

        Args:
            db_path: SQLite database file path
            max_size: Maximum number of cache entries (default 100,000)
            ttl: Time-to-live for cache entries in seconds (default 100 days)
        """
        self.db_path = db_path
        self.max_size = max_size
        self.ttl = ttl
        
        # Use async locks instead of thread locks
        self.lock = asyncio.Lock()
        self.cleanup_lock = asyncio.Lock()
        self.is_cleaning = False
        self.cleanup_interval = 500  # Increase cleanup interval to reduce cleanup frequency
        self.operations_since_cleanup = 0
        self.last_cleanup_time = time.time()
        
        # Cache hit rate statistics (using atomic operations)
        self.total_query_count = 0
        self.cache_hit_count = 0
        
        # Delayed update queue (to reduce write operations)
        self.update_queue = asyncio.Queue(maxsize=1000)
        self.update_task = None
        
        logger.info(f"AsyncWebSearchCache initialized: max_size={max_size}, ttl={ttl}s, db_path={db_path}")

    async def initialize(self):
        """Async database initialization"""
        await self._init_db()
        await self._cleanup()
        # Start background update task
        self.update_task = asyncio.create_task(self._process_update_queue())
        logger.info("AsyncWebSearchCache async initialization completed")

    async def shutdown(self):
        # Stop background update task
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass

    async def _init_db(self):
        """Initialize SQLite database, create cache table and indexes"""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute('''
            CREATE TABLE IF NOT EXISTS web_search_cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                search_result TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                last_accessed INTEGER NOT NULL
            )
            ''')
            
            # Create indexes
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON web_search_cache(last_accessed)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON web_search_cache(created_at)')
            
            # Enable WAL mode to improve concurrent performance
            await conn.execute('PRAGMA journal_mode=WAL')
            await conn.execute('PRAGMA synchronous=NORMAL')
            await conn.execute('PRAGMA cache_size=-64000')  # 64MB cache
            await conn.execute('PRAGMA temp_store=MEMORY')
            
            await conn.commit()

    def _get_query_hash(self, query: str) -> str:
        """Generate MD5 hash of search query"""
        return hashlib.md5(query.strip().encode('utf-8')).hexdigest()

    async def get(self, query: str) -> Optional[str]:
        """
        Get search results for a query from cache

        Args:
            query: Search query string

        Returns:
            Search results if cache hit and not expired; otherwise None
        """
        query = query.strip()
        if not query:
            return None

        query_hash = self._get_query_hash(query)
        current_time = int(time.time())
        self.total_query_count += 1

        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Set connection to WAL mode
                await conn.execute('PRAGMA journal_mode=WAL')
                
                cursor = await conn.execute('''
                    SELECT search_result, created_at 
                    FROM web_search_cache 
                    WHERE query_hash = ?
                ''', (query_hash,))
                result = await cursor.fetchone()

                if result:
                    search_result, created_at = result
                    # Check if cache is expired
                    if current_time - created_at <= self.ttl:
                        # Async update last accessed time (non-blocking read)
                        try:
                            self.update_queue.put_nowait((query_hash, current_time))
                        except asyncio.QueueFull:
                            pass  # Skip if queue is full, doesn't affect read performance
                        
                        # Count hits
                        self.cache_hit_count += 1
                        hit_rate = (self.cache_hit_count / self.total_query_count) * 100
                        logger.info(f"[CACHE HIT] Query: {query[:50]}... | Hit Rate: {hit_rate:.2f}% ({self.cache_hit_count}/{self.total_query_count})")
                        return search_result
                    else:
                        # Cache expired: async delete (non-blocking)
                        asyncio.create_task(self._delete_expired(query_hash))
                        logger.info(f"[CACHE EXPIRED] Query: {query[:50]}...")
                else:
                    # Cache miss
                    hit_rate = (self.cache_hit_count / self.total_query_count) * 100 if self.total_query_count > 0 else 0.0
                    logger.info(f"[CACHE MISS] Query: {query[:50]}... | Hit Rate: {hit_rate:.2f}% ({self.cache_hit_count}/{self.total_query_count})")
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
        
        return None

    async def _delete_expired(self, query_hash: str):
        """Async delete expired entries"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('PRAGMA journal_mode=WAL')
                await conn.execute('DELETE FROM web_search_cache WHERE query_hash = ?', (query_hash,))
                await conn.commit()
        except Exception as e:
            logger.error(f"Error deleting expired entry: {e}")

    async def _process_update_queue(self):
        """Background task: Batch process last_accessed updates"""
        batch = []
        batch_size = 50  # Update 50 entries per batch
        last_update_time = time.time()
        
        while True:
            try:
                # Wait for update request or timeout
                try:
                    item = await asyncio.wait_for(self.update_queue.get(), timeout=1.0)
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass
                
                # Execute batch update when batch size reached or 1 second passed since last update
                current_time = time.time()
                if len(batch) >= batch_size or (batch and current_time - last_update_time >= 1.0):
                    await self._batch_update_access_time(batch)
                    batch = []
                    last_update_time = current_time
                    
            except asyncio.CancelledError:
                # Flush before exit
                if batch:
                    try:
                        await self._batch_update_access_time(batch)
                    except Exception:
                        pass
                raise

            except Exception as e:
                logger.error(f"Error in update queue processor: {e}")
                await asyncio.sleep(1)

    async def _batch_update_access_time(self, batch: List[tuple]):
        """Batch update last_accessed time"""
        if not batch:
            return
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('PRAGMA journal_mode=WAL')
                
                # Use transaction for batch update
                await conn.execute('BEGIN TRANSACTION')
                for query_hash, access_time in batch:
                    await conn.execute('''
                        UPDATE web_search_cache 
                        SET last_accessed = ? 
                        WHERE query_hash = ?
                    ''', (access_time, query_hash))
                await conn.commit()
                
        except Exception as e:
            logger.error(f"Error in batch update: {e}")

    async def set(self, query: str, search_result: str):
        """
        Store search query and its results in cache

        Args:
            query: Search query string
            search_result: Search results
        """
        query = query.strip()
        if not query or not search_result:
            return

        query_hash = self._get_query_hash(query)
        current_time = int(time.time())

        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('PRAGMA journal_mode=WAL')
                
                await conn.execute('''
                    INSERT OR REPLACE INTO web_search_cache 
                    (query_hash, query, search_result, created_at, last_accessed) 
                    VALUES (?, ?, ?, ?, ?)
                ''', (query_hash, query, search_result, current_time, current_time))
                
                await conn.commit()
                logger.info(f"[CACHE STORED] Query: {query[:50]}...")
        except Exception as e:
            logger.error(f"Error storing to cache: {e}")

        # Check if cleanup needs to be triggered
        self.operations_since_cleanup += 1
        current_time_now = time.time()
        if (self.operations_since_cleanup >= self.cleanup_interval or
                current_time_now - self.last_cleanup_time > 360000):
            asyncio.create_task(self._trigger_async_cleanup())
            self.operations_since_cleanup = 0
            self.last_cleanup_time = current_time_now

    async def _trigger_async_cleanup(self):
        """Trigger async cleanup"""
        async with self.cleanup_lock:
            if self.is_cleaning:
                return
            self.is_cleaning = True

        try:
            await self._cleanup()
        except Exception as e:
            logger.error(f"Failed to execute async cleanup: {str(e)}", exc_info=True)
        finally:
            async with self.cleanup_lock:
                self.is_cleaning = False

    async def _cleanup(self):
        """Core cleanup logic: Delete expired entries and excess entries"""
        current_time = int(time.time())
        logger.info(f"Starting cache cleanup...")

        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('PRAGMA journal_mode=WAL')
                
                # 1. Delete expired entries
                cursor = await conn.execute('''
                    DELETE FROM web_search_cache 
                    WHERE created_at < ?
                ''', (current_time - self.ttl,))
                expired_count = cursor.rowcount

                # 2. Check current cache size
                cursor = await conn.execute('SELECT COUNT(*) FROM web_search_cache')
                result = await cursor.fetchone()
                current_size = result[0]
                
                lru_count = 0
                if current_size > self.max_size:
                    delete_count = current_size - self.max_size
                    await conn.execute('''
                        DELETE FROM web_search_cache 
                        WHERE query_hash IN (
                            SELECT query_hash 
                            FROM web_search_cache 
                            ORDER BY last_accessed ASC 
                            LIMIT ?
                        )
                    ''', (delete_count,))
                    lru_count = delete_count

                await conn.commit()
                
                logger.info(f"Cache cleanup completed: Expired={expired_count}, LRU={lru_count}, Size={current_size - expired_count - lru_count}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def clear(self):
        """Clear entire cache"""
        async with self.lock:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('PRAGMA journal_mode=WAL')
                await conn.execute('DELETE FROM web_search_cache')
                await conn.commit()
        
        self.total_query_count = 0
        self.cache_hit_count = 0
        logger.warning("AsyncWebSearchCache cleared")

    async def get_stats(self) -> Dict[str, Any]:
        """Get current cache status statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('PRAGMA journal_mode=WAL')
                
                cursor = await conn.execute('SELECT COUNT(*) FROM web_search_cache')
                result = await cursor.fetchone()
                total_entries = result[0]

                cursor = await conn.execute('SELECT MIN(created_at), MAX(created_at) FROM web_search_cache')
                result = await cursor.fetchone()
                min_create_time, max_create_time = result

                oldest_entry = datetime.fromtimestamp(min_create_time).strftime('%Y-%m-%d %H:%M:%S') if min_create_time else "N/A"
                newest_entry = datetime.fromtimestamp(max_create_time).strftime('%Y-%m-%d %H:%M:%S') if max_create_time else "N/A"

                hit_rate = (self.cache_hit_count / self.total_query_count) * 100 if self.total_query_count > 0 else 0.0

                return {
                    "total_entries": total_entries,
                    "max_size": self.max_size,
                    "ttl_seconds": self.ttl,
                    "oldest_entry_time": oldest_entry,
                    "newest_entry_time": newest_entry,
                    "total_query_count": self.total_query_count,
                    "cache_hit_count": self.cache_hit_count,
                    "cache_hit_rate": f"{hit_rate:.2f}%"
                }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


# --- Proxy Server --- #
class SerperProxyServer:
    """Proxy service class for handling Serper API requests with caching functionality."""
    def __init__(self):
        # Initialize cache (async initialization will be done in startup event)
        cache_size = int(os.environ.get("SERPER_CACHE_SIZE", "100000"))
        cache_ttl = int(os.environ.get("SERPER_CACHE_TTL", "8640000"))
        cache_path = os.environ.get("SERPER_CACHE_PATH", "serper_cache.db")
        self.cache = AsyncWebSearchCache(db_path=cache_path, max_size=cache_size, ttl=cache_ttl)

        server_host = os.environ.get("SERVER_HOST")
        crawl_page_port = os.environ.get("CRAWL_PAGE_PORT")
        self.crawl_page_endpoint = f"http:{server_host}:{crawl_page_port}/crawl_page" if server_host and crawl_page_port else None
        self.api_key_list = os.environ.get("WEB_SEARCH_SERPER_API_KEY","").split('|')
        assert self.api_key_list != [] , "No api keys configured."
        self.serpapi_base_url = os.environ.get("SERPAPI_BASE_URL","https://google.serper.dev/search")  # Fallback to official endpoint
        assert self.serpapi_base_url != "", "No serpapi_base_url configured."

        # For hit rate logging
        self.total_requests = 0
        self.cache_hits = 0
        self._stats_lock = asyncio.Lock()
        self.history = []
        self.max_history_length = 1000  # New: Limit history length to avoid memory leaks

        # default using first api key
        self._last_api_key_index = 0
        
        # HTTP client connection pool (increase concurrent connections)
        self.http_client = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAGES)  # New: Concurrency control semaphore

    async def initialize(self):
        """Async initialization"""
        await self.cache.initialize()
        # Create persistent HTTP client connection pool (increase concurrency)
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(SERPER_TIMEOUT),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50)
        )
        logger.info(f"SerperProxyServer initialized with HTTP connection pool (max concurrent pages: {MAX_CONCURRENT_PAGES})")

    async def shutdown(self):
        if self.http_client:
            await self.http_client.aclose()
        await self.cache.shutdown()
        logger.info("SerperProxyServer shutdown")

    def _generate_cache_key(self, payload: dict) -> str:
        """Generate cache key from request payload (includes num parameter to ensure different num values have independent cache)"""
        sorted_payload = json.dumps(payload, sort_keys=True)
        return f"serper:{sorted_payload}"

    async def _log_hit_rate(self, *, hit: bool):
        """Logs the cache hit rate."""
        async with self._stats_lock:
            self.total_requests += 1
            if hit:
                self.cache_hits += 1

            hit_rate = (
                (self.cache_hits / self.total_requests) * 100
                if self.total_requests > 0
                else 0
            )
            status = "HIT" if hit else "MISS"
            logger.info(
                f"Cache {status}. Rate: {hit_rate:.2f}% ({self.cache_hits}/{self.total_requests})"
            )

    def _get_header_case_insensitive(self, headers: dict, header_name: str) -> str:
        """Get HTTP header value case-insensitively"""
        header_name_lower = header_name.lower()
        for key, value in headers.items():
            if key.lower() == header_name_lower:
                return value
        return None

    # ========== Modification 1: Remove redundant prompt text from formatted results ==========
    def _format_results_to_string(self, serper_json: Dict[str, Any], query: str) -> str:
        """Formats the Serper JSON result into a structured string (removed redundant prompts)"""
        if "organic" not in serper_json or not serper_json["organic"]:
            return f"No results found for query: '{query}'"

        web_snippets = []
        for idx, page in enumerate(serper_json["organic"], 1):
            title = page.get("title", "No Title")
            link = page.get("link", "#")
            date_published = f"\nDate published: {page['date']}" if "date" in page else ""
            source = f"\nSource: {page.get('source', '')}" if "source" in page else ""
            snippet = f"\n{page.get('snippet', '')}".replace("Your browser can't play this video.", "")

            formatted_entry = (
                f"{idx}. [{title}]({link})"
                f"{date_published}{source}"
                f"\n{link}{snippet}"
            )
            web_snippets.append(formatted_entry.strip())
        
        num_results = len(web_snippets)
        # Remove original redundant prompts, keep only core results
        result_text = (
            f"Results for '{query}':\n\n"
            + "\n\n".join(web_snippets)
        )
        return result_text
    # ===========================================================

    # ========== Modification 2: Remove redundant prompts from _to_contents_multiqueries ==========
    def _to_contents_multiqueries(self, search_results: dict):
        """Convert search results dictionary to string (removed redundant prompts)"""
        all_contents = []
        total_results = 0

        for query, snippets in search_results.items():
            if isinstance(snippets, str):
                all_contents.append(f"## Query: '{query}'\n{snippets}")
                continue
            elif isinstance(snippets, list):
                if snippets == []:
                    all_contents.append(f"No results found for '{query}'")
                    continue

            web_snippets = []
            idx = 1
            for search_info in snippets:
                if isinstance(search_info, dict):
                    title = search_info.get('title', 'No title')
                    link = search_info.get('link', '#')
                    date = search_info.get('date', '')
                    source = search_info.get('source', '')
                    snippet = search_info.get('snippet', 'No snippet available')

                    redacted_version = (
                        f"{idx}. [{title}]({link})"
                        f"{date}{source}\n"
                        f"{self._pre_visit(link)}{snippet}"
                    ).replace("Your browser can't play this video.", "")

                    web_snippets.append(redacted_version)
                    idx += 1

            num_results_for_query = len(web_snippets)
            # Remove original redundant prompts, keep only core results
            query_content = (
                    f"## Query: '{query}'\n"
                    f"{num_results_for_query} results:\n\n"
                    + "\n\n".join(web_snippets)
            )
            all_contents.append(query_content)
            total_results += num_results_for_query

        if total_results > 0:
            summary = f"# Search Summary\nTotal results: {total_results}\n\n"
            return summary + "\n\n".join(all_contents)
        return "\n\n".join(all_contents)
    # ================================================================

    # ========== Modification 3: Remove redundant prompts related to history ==========
    def _check_history(self, url_or_query):
        """Simplified: Only record history, do not return prompt text"""
        # Delete oldest record when exceeding max length
        if len(self.history) >= self.max_history_length:
            self.history.pop(0)
        self.history.append((url_or_query, time.time()))
        return ""

    def _pre_visit(self, url):
        """Simplified: Only record history, do not return prompt text"""
        # Delete oldest record when exceeding max length
        if len(self.history) >= self.max_history_length:
            self.history.pop(0)
        self.history.append((url, time.time()))
        return ""
    # ======================================================
    
    def _select_api_key(self):
        """Simplified: Use round-robin strategy to select API Key (more stable)"""
        self._last_api_key_index = (self._last_api_key_index + 1) % len(self.api_key_list)
        selected_key = self.api_key_list[self._last_api_key_index]
        logger.info(f"Selected API Key (index {self._last_api_key_index}): {selected_key[-5:]}")
        return selected_key
    
    async def _fetch_single_page(self, page_request, api_key, request_data, page_num):
        """
        New: Get results for a single page (for concurrent calls)
        """
        async with self.semaphore:  # Control concurrency
            try:
                api_headers = {"Content-Type": "application/json", 'X-API-KEY': api_key}
                response = await self.http_client.post(
                    self.serpapi_base_url,
                    json=page_request,
                    headers=api_headers,
                    timeout=request_data.get("timeout", SERPER_TIMEOUT),
                )
                response.raise_for_status()
                page_results = response.json()
                logger.info(f"Concurrently fetched page {page_num+1} successfully, returned {len(page_results.get('organic', []))} results")
                return page_results.get("organic", [])
            except Exception as e:
                logger.error(f"Failed to concurrently fetch page {page_num+1}: {str(e)}")
                return []  # Return empty list on single page failure, doesn't affect other pages
    
    async def _try_api_request_paginated(self, api_request_data, api_key, request_data):
        """
        Core optimization: Concurrent paginated requests to Serper API, significantly improves speed
        - Calculate total pages needed, concurrently request all pages at once
        - Control maximum concurrency (MAX_CONCURRENT_PAGES) to avoid rate limiting
        - Merge all page results, deduplicate and truncate to target count
        """
        target_num = min(max(int(api_request_data.get("num", 10)), 1), 100)  # Limit max 100 results
        total_pages = (target_num + 9) // 10  # Round up to calculate pages needed (20 results = 2 pages, 21 results = 3 pages)
        total_pages = min(total_pages, 10)  # Max 10 pages = 100 results
        
        logger.info(f"Starting concurrent Serper API requests: target {target_num} results, {total_pages} pages, max concurrency {MAX_CONCURRENT_PAGES} pages")
        
        # Generate request parameters for all pages
        page_tasks = []
        for page_num in range(total_pages):
            page_request = api_request_data.copy()
            page_request["num"] = 10  # Fixed 10 results per page
            page_request["start"] = page_num * 10  # Pagination start position
            # Create concurrent task
            page_tasks.append(self._fetch_single_page(page_request, api_key, request_data, page_num))
        
        # Execute all paginated requests concurrently
        start_time = time.time()
        all_page_results = await asyncio.gather(*page_tasks)
        logger.info(f"All paginated requests completed, took {time.time() - start_time:.2f} seconds")
        
        # Merge all page results and deduplicate (by link)
        all_organic = []
        seen_links = set()
        for page_organic in all_page_results:
            for item in page_organic:
                link = item.get("link")
                if link and link not in seen_links:
                    seen_links.add(link)
                    all_organic.append(item)
                    # Early termination: stop merging when target count is reached
                    if len(all_organic) >= target_num:
                        break
            if len(all_organic) >= target_num:
                break
        
        # Build final result (truncate to target count)
        final_results = {
            "organic": all_organic[:target_num],
            "searchParameters": api_request_data
        }
        
        query = api_request_data.get('q', '')
        filter_year = api_request_data.get('filter_year')
        
        # Simplified empty result prompt
        if not final_results["organic"]:
            if filter_year:
                return f"No results found for '{query}' (year={filter_year})"
            return f"No results found for query: '{query}'"
            
        logger.info(f"Concurrent paginated requests completed: target {target_num} results, actually got {len(final_results['organic'])} results")
        return final_results

    async def process_request(self, request_data: dict, headers: dict) -> dict:
        """Optimization: Fix num allocation logic and variable scope issues for multiple queries"""
        api_request_data = request_data.copy()
        querylist = [query.strip() for query in request_data['q'].split('|') if query.strip()]
        if len(querylist) == 0:
            error_messages = f"Query '{request_data['q']}' split failed! Please strictly follow the requirement: separate each query with '|'!"
            return error_messages

        search_results = {}
        total_results = 0
        serp_num = min(max(int(request_data.get("num", 10)), 1), 100)  # Limit total num to 100
        remaining_queries = len(querylist)
        remaining_serp = serp_num

        seen_set = set()
        task_params = []  # Store parameters for each query to fix variable scope issues
        
        # Process multiple queries concurrently
        tasks = []
        for q in querylist:
            # Evenly distribute num, last query gets remaining count
            current_serp = max(1, remaining_serp // remaining_queries)
            if remaining_queries == 1:
                current_serp = remaining_serp  # Last query gets all remaining count
            
            api_request_data_copy = api_request_data.copy()
            api_request_data_copy['q'] = q
            api_request_data_copy["num"] = current_serp
            tasks.append(self.process_request_single(api_request_data_copy, headers))
            task_params.append((q, current_serp))  # Record query and allocated num
            remaining_serp -= current_serp
            remaining_queries -= 1
        
        # Wait for all queries to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for (q, current_serp), snippets in zip(task_params, results):
            if isinstance(snippets, Exception):
                logger.error(f"Error searching for query '{q}': {str(snippets)}")
                search_results[q] = f"Error when querying '{q}': {str(snippets)}"
                continue
            
            unique_results_for_query = []
            if isinstance(snippets, str):
                unique_results_for_query = [snippets]
            else:
                for result in snippets:
                    if isinstance(result, dict) and 'link' in result:
                        if result['link'] not in seen_set:
                            seen_set.add(result['link'])
                            unique_results_for_query.append(result)
            
            search_results[q] = unique_results_for_query[:current_serp]
            total_results += len(search_results[q])

        content = self._to_contents_multiqueries(search_results)
        return content

    async def process_request_single(self, request_data: dict, headers: dict) -> dict:
        """Process single Serper API request, prioritize cache, if cache miss then use concurrent paginated requests"""
        start_time = time.time()

        original_num = min(max(int(request_data.get("num", 10)), 1), 100)
        api_request_data = request_data.copy()

        try:
            cache_key = self._generate_cache_key(api_request_data)

            # Check cache
            cached_result_str = await self.cache.get(cache_key)
            results = None
            # If cache hit but cached result count is insufficient, re-request
            if cached_result_str:
                await self._log_hit_rate(hit=True)
                cached_result = json.loads(cached_result_str)
                cached_organic_len = len(cached_result.get("organic", []))
                if cached_organic_len < original_num:
                    # Cached result count insufficient, abandon cache and re-request
                    logger.info(f"Cached result count ({cached_organic_len}) < requested count ({original_num}), re-requesting Serper API")
                    cached_result_str = None
                else:
                    # Truncate to requested count
                    cached_result["organic"] = cached_result["organic"][:original_num]
                    results = cached_result
                    logger.info(f"Request processed from cache in {time.time() - start_time:.2f} seconds.")

            if not cached_result_str:
                # Cache miss, use concurrent paginated requests to Serper API
                await self._log_hit_rate(hit=False)
                logger.info(f"Forwarding to Serper API (concurrent pagination) for key: {cache_key}")

                if not self.api_key_list:
                    raise HTTPException(
                        status_code=401, detail="Serper API key not configured"
                    )
                
                api_key = self._select_api_key()
                try_times = 1  # Initially tried 1 time
                last_error = None

                # Try using current API Key for concurrent paginated requests
                try:
                    results = await self._try_api_request_paginated(api_request_data, api_key, request_data)
                    if isinstance(results, dict):
                        await self.cache.set(cache_key, json.dumps(results))
                        logger.info(f"Successfully cached concurrent paginated results for {request_data['q']} (num={original_num})")
                except Exception as first_error:
                    logger.warning(f"API Key ending {api_key[-5:]} failed, trying alternatives")
                    last_error = first_error
                    # Try remaining API Keys
                    while try_times < len(self.api_key_list):
                        try_times += 1
                        api_key = self._select_api_key()
                        try:
                            results = await self._try_api_request_paginated(api_request_data, api_key, request_data)
                            if isinstance(results, dict):
                                await self.cache.set(cache_key, json.dumps(results))
                                logger.info(f"Successfully cached concurrent paginated results for {request_data['q']} (num={original_num})")
                            logger.info(f"Successfully used alternative API Key ending {api_key[-5:]}")
                            break
                        except Exception as e:
                            logger.warning(f"API Key ending {api_key[-5:]} failed (attempt {try_times})")
                            last_error = e
                    else:
                        logger.error(f"All {len(self.api_key_list)} API Keys failed")
                        raise last_error

            # Format return results
            web_snippets: List[str] = list()
            idx = 0
            if isinstance(results, dict) and 'organic' in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    _search_result = {
                        "idx": idx,
                        "title": page["title"],
                        "date": date_published,
                        "snippet": snippet,
                        "source": source,
                        "link": page['link']
                    }

                    web_snippets.append(_search_result)
            elif isinstance(results, str):
                web_snippets.append(results)
                
            logger.info(f"Single query processed: {request_data['q'][:50]}... | Returned {len(web_snippets)} results | Total time: {time.time() - start_time:.2f}s")
            return web_snippets

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error calling Serper API: {e.response.status_code} {e.response.text}"
            )
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logger.error(f"Unexpected error during Serper API request: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"An unexpected error occurred: {str(e)}"
            )


# --- FastAPI Application Setup --- #

# Global proxy server instance
proxy_server = SerperProxyServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Initialize on startup
    await proxy_server.initialize()
    logger.info("Application started successfully")
    
    yield
    
    # Cleanup resources on shutdown
    await proxy_server.shutdown()
    logger.info("Application shutdown successfully")


# Create FastAPI application with new lifespan handler
app = FastAPI(
    title="Serper API Proxy with Async Cache & Concurrent Pagination",
    lifespan=lifespan
)


@app.post("/search")
async def serper_proxy_endpoint(request: Request):
    """Proxy endpoint for Serper API search (supports concurrent pagination to get >10 results, significantly improves speed)."""
    try:
        request_data = await request.json()
        headers = dict(request.headers)

        return await proxy_server.process_request(request_data, headers)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body.")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Unhandled exception in endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    cache_stats = await proxy_server.cache.get_stats()
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "cache_stats": {
            "total_requests": proxy_server.total_requests,
            "cache_hits": proxy_server.cache_hits,
            "hit_rate": (proxy_server.cache_hits / proxy_server.total_requests * 100) if proxy_server.total_requests > 0 else 0,
            **cache_stats
        },
        "concurrent_config": {
            "max_concurrent_pages": MAX_CONCURRENT_PAGES
        }
    }


@app.post("/test_single_query")
async def test_single_query(request: Request):
    """Test single query returning more than 10 results (concurrent pagination feature verification)"""
    request_data = await request.json()
    # Force single query, disable splitting
    request_data['q'] = request_data['q'].split('|')[0].strip()
    request_data['num'] = min(max(int(request_data.get('num', 20)), 1), 100)
    
    # Record time taken
    start_time = time.time()
    result = await proxy_server.process_request_single(request_data, dict(request.headers))
    total_time = time.time() - start_time
    
    return {
        "requested_num": request_data['num'],
        "returned_count": len(result) if isinstance(result, list) else 0,
        "total_time_seconds": round(total_time, 2),
        "results": result
    }


# --- Main Program Entry --- #
if __name__ == "__main__":
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = os.getenv("WEB_SEARCH_PORT", None)
    if port is None:
        raise NotImplementedError("[ERROR] WEBSEARCH_PORT NOT SET")
    port = int(port)

    logger.info(f"Starting Serper cache server with concurrent pagination support... http://{host}:{port}")
    logger.info(f"Supports single query returning up to 100 results (concurrent requests, max concurrency {MAX_CONCURRENT_PAGES} pages)")
    
    # Single process mode (development/Windows)
    # For multiple workers, use command line: 
    # uvicorn cached_serper_server_optimized:app --host 0.0.0.0 --port 8000 --workers 4
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info"
    )