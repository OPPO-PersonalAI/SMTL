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

import random
import json
import os
import requests

def WebSearchTool(web_search_url, query, topk=5):
    web_search_query = query
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": "",
    }
    payload = {
        "q": query,
        "num": topk,
    }
    try:
        response = requests.post(web_search_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, str):
            formatted_result = result
        elif isinstance(result, dict):
            formatted_result = []
            for idx, item in enumerate(result["organic"]):
                item["title"] = item.get("title", "No title")
                item["date"] = f"\nDate published: {item['date']}" if "date" in item else ""
                item["snippet"] = f"\n{item.get('snippet', 'No snippet')}"
                item["source"] = f"\nSource: {item.get('source', 'Unknown source')}"
                item["link"] = item.get('link', '#')
                item["idx"] = idx + 1
                formatted_result.append(
                    f"{item['idx']}. [{item['title']}]({item['link']})"
                    f"{item['date']}{item['source']}\n"
                    f"   {item['snippet'].strip()}"
                )
            formatted_result = "\n\n".join(formatted_result) if formatted_result else "No search results found"
    except requests.exceptions.RequestException as e:
        formatted_result = f"An error occurred: {e}"
    return formatted_result

def CrawlPageTool(crawl_page_url, api_key, api_url, model, query, url, content_max_len):
    url = [url]
    think_content = ""
    task = ""

    data = {
        "urls": url,
        "task": task, 
        "web_search_query": query,
        "think_content": think_content,
        "api_url": api_url,
        "api_key": api_key,
        "model": model,
        "summary_prompt_type": "webthinker_with_goal",
        "summary_type": "page",
        "content_max_len": content_max_len,
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(crawl_page_url, json=data, timeout=500, headers=headers)
    result = response.json()
    if result.get("success"):
        crawl_page_result = result["obs"]
    else:
        crawl_page_result = result.get("error_message")
    return crawl_page_result