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

import hashlib
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List, Union

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
import random

import aiohttp
import aiosqlite
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# ========== New: Import tokenizer related libraries ==========
import tiktoken  # For token counting
# ==============================================

from dotenv import load_dotenv
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CRAWL_PAGE_TIMEOUT = 500

# ========== New: Global token-related configuration ==========
# Maximum model context length (set based on error messages)
MAX_MODEL_CONTEXT_TOKENS = 120000
# Reserved tokens (for system prompt and response)
RESERVED_TOKENS = 4096
# Maximum tokens available for user prompt
MAX_PROMPT_TOKENS = MAX_MODEL_CONTEXT_TOKENS - RESERVED_TOKENS
# Token encoding used (compatible with most open-source models)
TOKEN_ENCODING = "cl100k_base"
# ==============================================

class CrawlPageRequest(BaseModel):
    """Request model"""
    urls: List[str] = Field(..., description="List of URLs to crawl")
    think_content: str = Field(..., description="Thinking content for guiding summarization, or generate click_intent based on think_content")
    web_search_query: str = Field(..., description="Web search query")
    summary_type: Optional[str] = Field("page", description="Summary type")
    summary_prompt_type: Optional[str] = Field("webthinker_with_goal", description="Summary prompt template type")

    # API configuration
    api_url: Optional[str] = Field(..., description="API URL")
    api_key: Optional[str] = Field(..., description="API key")
    model: Optional[str] = Field(..., description="Model name")

    # Parameters not currently used
    messages: Optional[List[Dict]] = Field(None, description="Message history (optional, for future use)")

class CrawlPageResponse(BaseModel):
    """Response model"""
    success: bool
    obs: str
    error_message: Optional[str] = None
    processing_time: float

# ========== New: Token utility class ==========
class TokenUtils:
    """Token counting and content truncation utility class"""
    def __init__(self, encoding_name: str = TOKEN_ENCODING):
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        """Calculate token count of text"""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to specified maximum token count
        Keep key content at the beginning and end, truncate the middle
        """
        if not text:
            return ""
        
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Calculate tokens to keep at front and back (60% front, 40% back)
        keep_front = int(max_tokens * 0.6)
        keep_back = max_tokens - keep_front
        
        # Truncate and decode
        truncated_tokens = tokens[:keep_front] + tokens[-keep_back:]
        truncated_text = self.encoding.decode(truncated_tokens)
        
        # Simplified truncation note
        truncation_note = f"\n\n[Content too long, truncated from {len(tokens)} tokens to {max_tokens} tokens]"
        return truncated_text + truncation_note

# Initialize token utility
token_utils = TokenUtils()
# ==============================================

class AsyncJinaCrawlCache:
    """High-performance LRU cache based on async SQLite for caching Jina crawled content"""
    
    def __init__(self, db_path: str = "jina_cache.db", max_size: int = 100000, ttl: int = 86400):
        """
        Initialize cache
        
        Args:
            db_path: SQLite database file path
            max_size: Maximum number of cache entries
            ttl: Time-to-live for cache entries in seconds, default is 1 day
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
        
        # Delayed update queue (to reduce write operations)
        self.update_queue = asyncio.Queue(maxsize=1000)
        self.update_task = None
        
        logger.info(f"AsyncJinaCrawlCache initialized: max_size={max_size}, ttl={ttl}s, db_path={db_path}")

    async def initialize(self):
        """Async database initialization"""
        await self._init_db()
        await self._cleanup()
        # Start background update task
        self.update_task = asyncio.create_task(self._process_update_queue())
        logger.info("AsyncJinaCrawlCache async initialization completed")
    
    async def _init_db(self):
        """Initialize SQLite database and table structure"""
        async with aiosqlite.connect(self.db_path) as conn:
            # Enable WAL mode to improve concurrent performance (must be set before any other operations)
            await conn.execute('PRAGMA journal_mode=WAL')
            await conn.execute('PRAGMA synchronous=NORMAL')
            await conn.execute('PRAGMA cache_size=-64000')  # 64MB cache
            await conn.execute('PRAGMA temp_store=MEMORY')
            
            # Create cache table
            await conn.execute('''
            CREATE TABLE IF NOT EXISTS jina_cache (
                url_hash TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                last_accessed INTEGER NOT NULL
            )
            ''')
            
            # Create statistics table (shared statistics across processes)
            await conn.execute('''
            CREATE TABLE IF NOT EXISTS cache_stats (
                stat_key TEXT PRIMARY KEY,
                stat_value INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            ''')
            
            # Initialize statistics
            current_time = int(time.time())
            await conn.execute('''
                INSERT OR IGNORE INTO cache_stats (stat_key, stat_value, updated_at)
                VALUES ('total_requests', 0, ?), ('cache_hits', 0, ?)
            ''', (current_time, current_time))
            
            # Create indexes
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON jina_cache(last_accessed)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON jina_cache(created_at)')
            
            await conn.commit()
    
    def _get_url_hash(self, url: str) -> str:
        """Generate URL hash as cache key"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    async def _increment_stat(self, stat_key: str, increment: int = 1):
        """Atomically increment statistics (cross-process safe)"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('PRAGMA journal_mode=WAL')
                current_time = int(time.time())
                await conn.execute('''
                    UPDATE cache_stats 
                    SET stat_value = stat_value + ?, updated_at = ?
                    WHERE stat_key = ?
                ''', (increment, current_time, stat_key))
                await conn.commit()
        except Exception as e:
            logger.error(f"Error updating stat {stat_key}: {e}")
    
    async def _get_stats_from_db(self) -> tuple:
        """Get statistics from database (shared across processes)"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('PRAGMA journal_mode=WAL')
                
                cursor = await conn.execute('''
                    SELECT stat_key, stat_value 
                    FROM cache_stats 
                    WHERE stat_key IN ('total_requests', 'cache_hits')
                ''')
                stats = await cursor.fetchall()
                
                total_requests = 0
                cache_hits = 0
                for stat_key, stat_value in stats:
                    if stat_key == 'total_requests':
                        total_requests = stat_value
                    elif stat_key == 'cache_hits':
                        cache_hits = stat_value
                
                return total_requests, cache_hits
        except Exception as e:
            logger.error(f"Error reading stats from DB: {e}")
            return 0, 0
    
    async def get(self, url: str) -> Optional[str]:
        """
        Get content for URL from cache
        
        Args:
            url: URL to retrieve
            
        Returns:
            Cached content if cache hit and not expired; otherwise None
        """
        url_hash = self._get_url_hash(url)
        current_time = int(time.time())
        
        # Increment total request count (shared across processes)
        await self._increment_stat('total_requests', 1)
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Set connection to WAL mode
                await conn.execute('PRAGMA journal_mode=WAL')
                
                cursor = await conn.execute(
                    'SELECT content, created_at FROM jina_cache WHERE url_hash = ?', 
                    (url_hash,)
                )
                result = await cursor.fetchone()
                
                if result:
                    content, created_at = result
                    # Check if expired
                    if current_time - created_at <= self.ttl:
                        # Async update last accessed time (non-blocking read)
                        try:
                            self.update_queue.put_nowait((url_hash, current_time))
                        except asyncio.QueueFull:
                            pass  # Skip if queue is full, doesn't affect read performance
                        
                        # Increment hit count (shared across processes)
                        await self._increment_stat('cache_hits', 1)
                        
                        # Get latest statistics
                        total_requests, cache_hits = await self._get_stats_from_db()
                        hit_rate = (cache_hits / total_requests) * 100 if total_requests > 0 else 0.0
                        logger.info(f"[CACHE HIT] URL: {url[:80]}... | Hit Rate: {hit_rate:.2f}% ({cache_hits}/{total_requests})")
                        return content
                    else:
                        # Cache expired: async delete (non-blocking)
                        asyncio.create_task(self._delete_expired(url_hash))
                        logger.info(f"[CACHE EXPIRED] URL: {url[:80]}...")
                else:
                    # Cache miss
                    total_requests, cache_hits = await self._get_stats_from_db()
                    hit_rate = (cache_hits / total_requests) * 100 if total_requests > 0 else 0.0
                    logger.info(f"[CACHE MISS] URL: {url[:80]}... | Hit Rate: {hit_rate:.2f}% ({cache_hits}/{total_requests})")
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
        
        return None
    
    async def _delete_expired(self, url_hash: str):
        """Async delete expired entries"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('PRAGMA journal_mode=WAL')
                await conn.execute('DELETE FROM jina_cache WHERE url_hash = ?', (url_hash,))
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
                for url_hash, access_time in batch:
                    await conn.execute('''
                        UPDATE jina_cache 
                        SET last_accessed = ? 
                        WHERE url_hash = ?
                    ''', (access_time, url_hash))
                await conn.commit()
                
        except Exception as e:
            logger.error(f"Error in batch update: {e}")
    
    async def set(self, url: str, content: str):
        """
        Store URL and content in cache
        
        Args:
            url: URL to cache
            content: Content to cache
        """
        url_hash = self._get_url_hash(url)
        current_time = int(time.time())
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('PRAGMA journal_mode=WAL')
                
                await conn.execute('''
                    INSERT OR REPLACE INTO jina_cache 
                    (url_hash, url, content, created_at, last_accessed) 
                    VALUES (?, ?, ?, ?, ?)
                ''', (url_hash, url, content, current_time, current_time))
                
                await conn.commit()
                logger.info(f"[CACHE STORED] URL: {url[:80]}...")
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
                    DELETE FROM jina_cache 
                    WHERE created_at < ?
                ''', (current_time - self.ttl,))
                expired_count = cursor.rowcount

                # 2. Check current cache size
                cursor = await conn.execute('SELECT COUNT(*) FROM jina_cache')
                result = await cursor.fetchone()
                current_size = result[0]
                
                lru_count = 0
                if current_size > self.max_size:
                    delete_count = current_size - self.max_size
                    await conn.execute('''
                        DELETE FROM jina_cache 
                        WHERE url_hash IN (
                            SELECT url_hash 
                            FROM jina_cache 
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
                await conn.execute('DELETE FROM jina_cache')
                
                # Reset statistics
                current_time = int(time.time())
                await conn.execute('''
                    UPDATE cache_stats 
                    SET stat_value = 0, updated_at = ?
                    WHERE stat_key IN ('total_requests', 'cache_hits')
                ''', (current_time,))
                
                await conn.commit()
        
        logger.warning("AsyncJinaCrawlCache cleared")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get current cache status statistics (shared across processes)"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('PRAGMA journal_mode=WAL')
                
                cursor = await conn.execute('SELECT COUNT(*) FROM jina_cache')
                result = await cursor.fetchone()
                total_entries = result[0]
                
                cursor = await conn.execute('SELECT MIN(created_at), MAX(created_at) FROM jina_cache')
                result = await cursor.fetchone()
                min_create_time, max_create_time = result
                
                oldest_entry = datetime.fromtimestamp(min_create_time).strftime('%Y-%m-%d %H:%M:%S') if min_create_time else "N/A"
                newest_entry = datetime.fromtimestamp(max_create_time).strftime('%Y-%m-%d %H:%M:%S') if max_create_time else "N/A"
                
                # Read statistics from database (shared across processes)
                total_requests, cache_hits = await self._get_stats_from_db()
                hit_rate = (cache_hits / total_requests) * 100 if total_requests > 0 else 0.0
                
                return {
                    "total_entries": total_entries,
                    "max_size": self.max_size,
                    "ttl_seconds": self.ttl,
                    "oldest_entry_time": oldest_entry,
                    "newest_entry_time": newest_entry,
                    "total_request_count": total_requests,
                    "cache_hit_count": cache_hits,
                    "cache_hit_rate": f"{hit_rate:.2f}%"
                }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


class CrawlPageServer:
    def __init__(self):
        logger.info("Initializing CrawlPageServer")
        self.jina_timeout = 30
        self.summary_timeout = 300
        self.max_retries = 5
        self.jina_token_budget = 200000
        self.api_key_list = os.environ.get("JINA_API_KEY","").split("|")
        logger.info(f"API_KEY_LIST: {self.api_key_list}")
        assert self.api_key_list != [], "No api key configured."
        
        # Initialize cache (async initialization will be done in startup event)
        cache_size = int(os.environ.get("JINA_CACHE_SIZE", "100000"))
        cache_ttl = int(os.environ.get("JINA_CACHE_TTL", "8640000"))  # Default 100 days
        cache_path = os.environ.get("JINA_CACHE_PATH", "jina_cache.db")
        self.cache = AsyncJinaCrawlCache(db_path=cache_path, max_size=cache_size, ttl=cache_ttl)
        
        logger.info(f"CrawlPageServer initialized with jina_timeout={self.jina_timeout}s, summary_timeout={self.summary_timeout}s, token_budget={self.jina_token_budget}")
    
    async def initialize(self):
        """Async initialization"""
        await self.cache.initialize()
        logger.info("CrawlPageServer initialized successfully")
    
    def _select_api_key_random(self):
        """Select an API Key using random strategy"""
        return random.randint(0, len(self.api_key_list) - 1)

    def _select_api_key_with_round_robin(self, api_key_index):
        """Select API Key using round-robin strategy"""
        return (api_key_index + 1) % len(self.api_key_list)
    
    async def _fetch_with_api(self, session: aiohttp.ClientSession, url: str, base_delay: float = 1.0, max_delay: float = 16.0) -> Tuple[str, str]:
        """
        Fetch a single URL using API with round-robin.
        First check cache, if cache hit return cached content directly.
        Each request timeout = timeout seconds.
        Return (content, url) immediately on success; return (error_msg, url) if all attempts fail.
        """
        # Check cache first
        cached_content = await self.cache.get(url)
        if cached_content:
            logger.info(f"Cache hit for URL: {url}")
            return cached_content, url
        
        # Cache miss, use load balancing to select an API Key
        api_key_index = self._select_api_key_random()
        logger.info(f"Cache miss for URL: {url}. Choosing api_key: {self.api_key_list[api_key_index]}")
        try_times = 0
        try:
            try_times += 1
            results = await self._fetch_with_retry(session, url, base_delay, max_delay, self.api_key_list[api_key_index])
            if isinstance(results,tuple) and "[Page content not accessible" in results[0]:
                raise Exception("Unsuccessful crawl")
            logger.info(f"Successfully crawled page with api_key ending with {self.api_key_list[api_key_index][-5:]}")
        except Exception as first_error:
            logger.warning(f"API Key with ending {self.api_key_list[api_key_index][-5:]} failed, trying alternatives")

            if try_times == len(self.api_key_list):
                # No other API Keys available
                logger.warning(f"API Key with ending {self.api_key_list[api_key_index][-5:]} failed, no alternatives left")
                raise first_error
            
            # Try remaining API Keys
            last_error = first_error
            while try_times < len(self.api_key_list):
                try:
                    try_times += 1
                    api_key_index = self._select_api_key_with_round_robin(api_key_index)
                    logger.info(f"Choosing api_key: {self.api_key_list[api_key_index]}")
                    results = await self._fetch_with_retry(session, url, base_delay, max_delay, self.api_key_list[api_key_index])
                    if isinstance(results,tuple) and "[Page content not accessible" in results[0]:
                        raise Exception("Unsuccessful crawl")
                    logger.info(f"Successfully crawled page with api_key ending with {self.api_key_list[api_key_index][-5:]}")
                    break
                except Exception as e:
                    logger.warning(f"API Key with ending {self.api_key_list[api_key_index][-5:]} failed, trying alternatives")
                    last_error = e
            else:
                # All API Keys failed
                raise last_error
        
        # Successfully fetched content, store in cache
        if isinstance(results, tuple) and not results[0].startswith("[Page content not accessible"):
            await self.cache.set(url, results[0])
        
        return results
    
    async def _fetch_with_retry(self, session: aiohttp.ClientSession, url: str, base_delay: float = 1.0, max_delay: float = 16.0, apikey: str = '') -> Tuple[str, str]:
        """
        Fetch a single URL with at most self.max_retries attempts.
        Each request timeout = timeout seconds.
        Return (content, url) immediately on success; return (error_msg, url) if all attempts fail.
        """
        assert apikey != "", "No api key when fetching."
        attempt = 0
        last_exc = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                logger.info(f"[Attempt {attempt}/{self.max_retries}] {url}")
                timeout = aiohttp.ClientTimeout(total=self.jina_timeout)
                jina_url = f"https://r.jina.ai/{url}"
                headers = {
                    'Authorization': f'Bearer {apikey}',
                    'X-Engine': 'browser',
                    'X-Return-Format': 'text',
                    "X-Remove-Selector": "header, .class, #id",
                    'X-Timeout': str(self.jina_timeout),
                    "X-Retain-Images": "none",
                    'X-Token-Budget': "200000"
                }

                async with session.get(jina_url, headers=headers, timeout=timeout) as resp:
                    resp.raise_for_status()
                    content = await resp.text()
                    return content, url
            except asyncio.TimeoutError:
                last_exc = f"Timeout after {self.jina_timeout}s"
                logger.warning(f"[Attempt {attempt}] Timeout for {url}")
            except Exception as e:
                last_exc = str(e)
                logger.warning(f"[Attempt {attempt}] Error for {url}: {e}")

            if attempt < self.max_retries:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                await asyncio.sleep(delay)

        # ========== Modification 1: Simplify crawl failure prompt ==========
        return f"Page content not accessible: {last_exc}", url
        # ============================================

    async def read_page_async(self, session: aiohttp.ClientSession, url: str) -> Tuple[str, str]:
        return await self._fetch_with_api(session, url)

    def validate_urls(self, urls: List[str]) -> List[str]:
        """Validate HTTP/HTTPS URLs."""
        processed_urls = []
        for url in urls:
            url = url.strip()
            if not url:
                continue
            if url.startswith(('http://', 'https://')):
                processed_urls.append(url)
            else:
                logger.warning(f"Invalid URL format (must start with http:// or https://): {url}")        
        return processed_urls
    
    def get_click_intent_instruction(self, prev_reasoning: str) -> str:
        return f"""Based on the previous thoughts below, provide the detailed intent of the latest click action.
    Previous thoughts: {prev_reasoning}
    Please provide the current click intent."""

    # ========== Modification: Add token truncation logic ==========
    def get_summary_prompt_new(self, url: str, query: str, content: str) -> str:
        """Generate prompt for content summarization with token limit protection"""
        
        # 1. Simplify prompt template, remove redundant instructions
        template = f"""Process the following webpage content and extract information relevant to the goal:

## Webpage URL
{url}

## Webpage Content 
{{content_placeholder}}

## Information Goal
{query}

## Task Guidelines
1. Locate specific sections/data directly related to the goal
2. Extract the most relevant information, keep full original context as much as possible
3. Organize findings into a concise and comprehensive summary

**Output JSON with "rationale", "evidence", "summary" fields**
""".strip()
        
        # Calculate token count of template (replace placeholder with empty string)
        template_tokens = token_utils.count_tokens(template.replace("{content_placeholder}", ""))
        
        # 2. Calculate maximum tokens available for content
        max_content_tokens = MAX_PROMPT_TOKENS - template_tokens
        if max_content_tokens <= 0:
            max_content_tokens = 1000  # Fallback value
        
        # 3. Truncate content to maximum token count
        truncated_content = token_utils.truncate_text_to_tokens(content, max_content_tokens)
        
        # 4. Build final prompt
        final_prompt = template.replace("{content_placeholder}", truncated_content)
        
        # 5. Log token usage
        total_tokens = token_utils.count_tokens(final_prompt)
        logger.info(f"Prompt token count: {total_tokens}/{MAX_PROMPT_TOKENS} (template: {template_tokens}, content: {token_utils.count_tokens(truncated_content)})")
        
        return final_prompt
    # ==============================================

    async def call_ai_api_async(self, system_prompt: str, user_prompt: str, api_url: str, api_key: str, model: str, max_retries: int = 5, base_delay: float = 5) -> str:
        """Async call AI API with retry mechanism"""
        logger.info(f"Calling AI API with model: {model}, API URL: {api_url}, max_retries: {max_retries}")
        attempt = 0
        last_error = None

        while attempt < max_retries:
            attempt += 1
            try:
                logger.info(f"[Attempt {attempt}/{max_retries}] Calling AI API...")
                client = AsyncOpenAI(base_url=api_url, api_key=api_key)
                
                # ========== Modification: Add token check ==========
                # Calculate total token count (system + user prompt)
                total_prompt_tokens = token_utils.count_tokens(system_prompt) + token_utils.count_tokens(user_prompt)
                if total_prompt_tokens > MAX_PROMPT_TOKENS:
                    logger.warning(f"Prompt tokens {total_prompt_tokens} exceed limit {MAX_PROMPT_TOKENS}, truncating user prompt")
                    # Recalculate tokens available for user prompt
                    max_user_tokens = MAX_PROMPT_TOKENS - token_utils.count_tokens(system_prompt)
                    if max_user_tokens <= 0:
                        max_user_tokens = 1000
                    # Truncate user prompt
                    user_prompt = token_utils.truncate_text_to_tokens(user_prompt, max_user_tokens)
                # ==========================================
                
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_headers={
                        'X-DashScope-DataInspection':'{"input":"disable","output":"disable"}'  # Key parameter
                    },
                    stream=False,
                    timeout=self.summary_timeout
                )
                content = completion.choices[0].message.content
                logger.info(f"AI API response received, length: {len(content)} chars")
                return content
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[Attempt {attempt}] AI API call failed: {last_error}, API_KEY: {api_key}, API_URL: {api_url}")
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1)) # 60s -> 120s -> 240s -> 480s
                    logger.info(f"Waiting {delay:.2f}s before retry...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("All retry attempts exhausted, AI API call failed.")

        # ========== Modification 2: Simplify AI call failure prompt ==========
        return f"AI processing failed (retried {max_retries} times): {last_error}"
        # ==============================================
    
    async def summarize_content(self, content: str, request: CrawlPageRequest) -> str:
        """Helper function to summarize content"""
        logger.info(f"Summarizing content of length: {len(content)} chars, token count: {token_utils.count_tokens(content)}")

        url_single = request.urls[0]
        detailed_prompt = self.get_summary_prompt_new(
           url_single, request.web_search_query, content
        )
        return await self.call_ai_api_async(
            "You are a summary agent robot.", detailed_prompt,
            request.api_url, request.api_key, request.model
        )

    async def process_crawl_page(self, request: CrawlPageRequest) -> CrawlPageResponse:
        start_time = time.time()
        try:
            # Use parameters directly as passed
            logger.info("--------- Starting crawl_page request processing ---------")
            logger.info(f"Processing {len(request.urls)} URLs: {request.urls}")
            logger.info(f"web_search_query='{request.web_search_query}'")
            logger.info(f"Summary type: {request.summary_type}")
            
            # Validate and clean URL list
            urls = self.validate_urls(request.urls)
            if not urls:
                logger.warning("No valid URLs found after validation")
                # ========== Modification 3: Simplify invalid URL prompt ==========
                return CrawlPageResponse(
                    success=False,
                    obs="",
                    error_message="No valid URLs found. Please provide complete URLs starting with http:// or https:// and retry.",
                    processing_time=time.time() - start_time
                )
                # ============================================
            
            logger.info(f"Starting to process {len(urls)} URLs: {urls}")
            
            # Async fetch page content
            page_contents = ""
            logger.info("Creating aiohttp session for page fetching")
            async with aiohttp.ClientSession() as session:
                tasks = [self.read_page_async(session, url) for url in urls]
                logger.info(f"Fetching {len(tasks)} pages concurrently")
                page_results = await asyncio.gather(*tasks, return_exceptions=True)

                processed_results = []
                for i, result in enumerate(page_results):
                    if isinstance(result, Exception):
                        logger.error(f"Unhandled exception for URL {urls[i]}: {result}")
                        # ========== Modification 4: Simplify exception handling prompt ==========
                        processed_results.append((f"[Page content not accessible: {result}]", urls[i]))
                        # ============================================
                    else:
                        processed_results.append(result)
                page_results = processed_results

            ##### End Jina read page #####
            logger.info(f"Page fetching completed after {time.time() - start_time} seconds")
            
            ##### Start page summary #####
            summary_type = request.summary_type
            logger.info(f"Using summary type: {summary_type}")
            if summary_type == "none":
                # No summarization, just concatenate all content
                logger.info("No summarization requested, concatenating raw content")
                summary_result = "\n\n".join(f"Page {i+1} [{result[1]}]: {result[0]}" for i, result in enumerate(page_results))
                logger.info(f"Combined content length: {len(summary_result)} chars, token count: {token_utils.count_tokens(summary_result)}")

            elif summary_type == "once":
                # Single summarization of all content
                logger.info("Using 'once' strategy - single summarization of all content")
                page_contents = "\n\n".join(f"Page {i+1} [{result[1]}]: {result[0]}" for i, result in enumerate(page_results))
                logger.info(f"Combined content for summarization: {len(page_contents)} chars, token count: {token_utils.count_tokens(page_contents)}")
                summary_result = await self.summarize_content(page_contents, request)

            elif summary_type == "page":
                # Page-by-page summarization
                logger.info(f"Using page summary strategy")
                
                # Process all pages concurrently
                page_tasks = []
                page_indices = []  # Record which pages need summary
                for i, (content, url) in enumerate(page_results):
                    logger.info(f"Creating task for page {i+1}/{len(page_results)}, URL: {url}, content length: {len(content) / 1000:.2f}k characters, token count: {token_utils.count_tokens(content)}")
                    if content.startswith("[Page content not accessible:"):
                        # For inaccessible pages, don't create task, process directly later
                        page_indices.append((i, False))  # False means no summary needed
                    else:
                        # Create summary task
                        task = self.summarize_content(content, request)
                        page_tasks.append(task)
                        page_indices.append((i, True))  # True means summary needed
                
                # Wait for all page summaries to complete
                if page_tasks:
                    logger.info(f"Processing {len(page_tasks)} pages concurrently")
                    page_results_summary = await asyncio.gather(*page_tasks, return_exceptions=True)
                else:
                    page_results_summary = []
                
                # Process results
                page_summaries = []
                summary_idx = 0
                for i, needs_summary in page_indices:
                    content, url = page_results[i]
                    if not needs_summary:
                        # Inaccessible pages, use original content directly
                        page_summaries.append(f"Page {i+1} [{url}]: {content}")
                    else:
                        # Get summary result
                        result = page_results_summary[summary_idx]
                        summary_idx += 1
                        if isinstance(result, Exception):
                            logger.error(f"Error processing page {i+1} [{url}]: {str(result)}")
                            error_msg = str(result)
                            # ========== Modification 5: Simplify page processing error prompt ==========
                            page_summaries.append(f"Page {i+1} [{url}] Summary:\n[Error: {error_msg}]")
                            # ==============================================
                        else:
                            page_summaries.append(f"Page {i+1} [{url}] Summary:\n{result}")

                summary_result = "\n\n".join(page_summaries)
            else:
                logger.error(f"Invalid summary_type: {summary_type}")
                raise ValueError(f"Invalid summary_type: {summary_type}, only support 'none', 'once', 'page'")
            
            processing_time = time.time() - start_time
            logger.info(f"crawl page done, cost time: {processing_time:.2f} seconds, result length: {len(summary_result)} chars, token count: {token_utils.count_tokens(summary_result)}")
            logger.info("--------- Request processing successful ---------")
            
            return CrawlPageResponse(
                success=True,
                obs=summary_result,
                processing_time=processing_time
            )
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"crawl page error: {str(e)}", exc_info=True)
            logger.error("--------- Request processing failed ---------")
            error_msg = str(e)
            error_msg_lower = error_msg.lower()
            # ========== Modification 6: Remove all redundant error suggestion prompts ==========
            # Keep only concise error information, remove all suggestion text
            return CrawlPageResponse(
                success=False,
                obs="",
                error_message=f"crawl page error: {error_msg}",
                processing_time=processing_time
            )
            # =====================================================

# Create server instance
crawl_server = CrawlPageServer()

# Create FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize on startup
    await crawl_server.initialize()
    logger.info("CrawlPage server started successfully")
    
    yield
    
    # Cleanup on shutdown
    logger.info("CrawlPage server shutdown")

app = FastAPI(
    title="CrawlPage Tool Server",
    description="FastAPI-based crawl_page tool service with high concurrency and fault tolerance",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return await crawl_server.cache.get_stats()

@app.post("/cache/clear")
async def clear_cache():
    """Clear cache"""
    await crawl_server.cache.clear()
    return {"status": "success", "message": "Cache cleared"}

@app.post("/crawl_page", response_model=CrawlPageResponse)
async def crawl_page_endpoint(request: CrawlPageRequest):
    logger.info(f"Received crawl_page request from client")
    """
    CrawlPage tool interface
    
    Parameters:
    # Parameters
    - urls: List[str] - Required, list of URLs to crawl
    - think_content: str - Required, thinking content for guiding summarization, or generate click_intent based on think_content
    - web_search_query: str - Required, WebSearch query
    - summary_type: str - Required, set to 'page'
    - summary_prompt_type: str - Required, set to 'webthinker_with_goal'

    # API KEY
    - api_url: str - Required, AI API URL
    - api_key: str - Required, AI API Key
    - model: str - Required, AI model
    
    # Parameters not currently needed
    - messages: Optional[List[Dict]] - Not currently needed. Message history (optional, not currently used)
    - task: str - Not currently needed. Task description

    Returns:
    response: CrawlPageResponse, contains four fields
    - success: bool - Whether successful
    - obs: str - AI summary
    - processing_time: float - Processing time
    - error_message: Optional[str] - Error message (if failed)
    """
    try:
        result = await asyncio.wait_for(
            crawl_server.process_crawl_page(request),
            timeout=CRAWL_PAGE_TIMEOUT
        )
        logger.info(f"Request completed successfully, success={result.success}")
        return result
    except asyncio.TimeoutError:
        logger.error(f"Request timeout after {CRAWL_PAGE_TIMEOUT}s")
        raise HTTPException(status_code=504, detail=f"Request timeout: {CRAWL_PAGE_TIMEOUT} seconds")
    except Exception as e:
        logger.error(f"Endpoint processing exception: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint (shared statistics across processes)"""
    cache_stats = await crawl_server.cache.get_stats()
    
    # Get global statistics from database (not single worker statistics)
    total_requests, cache_hits = await crawl_server.cache._get_stats_from_db()
    
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "cache_stats": {
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "hit_rate": (cache_hits / total_requests * 100) if total_requests > 0 else 0,
            **cache_stats
        },
        # ========== New: Add token configuration information ==========
        "token_config": {
            "max_model_context_tokens": MAX_MODEL_CONTEXT_TOKENS,
            "reserved_tokens": RESERVED_TOKENS,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "token_encoding": TOKEN_ENCODING
        }
        # ==============================================
    }

@app.get("/")
async def root():
    return {
        "message": "CrawlPage Tool Server",
        "version": "1.0.0",
        "endpoints": {
            "crawl_page": "/crawl_page",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    # Get host and port from environment variables or default values
    host = os.getenv("SERVER_HOST", None)
    port = os.getenv("CRAWL_PAGE_PORT", None)
    if port == None:
        raise NotImplementedError("[ERROR] CRAWL_PAGE_PORT NOT SET!")
    port = int(port)

    # Run server
    logger.info(f"Configuring server with host={host}, port={port}, workers=10")
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        workers=20,  # Support multiple processes to improve concurrency
        reload=False,
        access_log=True,
        log_level="info"
    ) 
    server = uvicorn.Server(config)
    try:
        logger.info(f"Starting CrawlPage server... http://{host}:{port}")
        server.run()
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise