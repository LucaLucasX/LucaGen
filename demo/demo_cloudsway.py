import os
import json
import requests
from urllib.parse import quote
from typing import Optional

from dotenv import load_dotenv
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, RetryError
)


# 自定义错误类型
class RateLimitError(Exception):
    pass


class RetryableError(Exception):
    pass


class CloudswaySearchClient:
    """Cloudsway Search Client"""

    FULLTEXT_URL = "https://searchapi.cloudsway.net/search/NbYyRVhrORhcVYNm/full"

    def __init__(self):
        # 读取 .env 文件
        load_dotenv()
        self.api_key = os.getenv("CLOUDSWAY_API_KEY")
        if not self.api_key:
            raise ValueError("CLOUDSWAY_API_KEY environment variable is required")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        # 使用 requests.Session 复用连接
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def close(self):
        """关闭 Session"""
        self.session.close()

    # ---------------------------
    # 同步重试版请求
    # ---------------------------
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, RetryableError)),
        reraise=True,
    )
    def _search_with_retry(self, url, params, query, include_raw_content):
        try:
            resp = self.session.get(url, params=params, timeout=30)

            if resp.status_code == 429:
                raise RateLimitError("Rate limit")

            if resp.status_code >= 500:
                raise RetryableError(f"Server {resp.status_code}")

            resp.raise_for_status()
            data = resp.json()

            request_id = resp.headers.get("x-ws-request-id")
            return self._transform_to_tavily_format(data, query, request_id, include_raw_content)

        except requests.exceptions.Timeout as e:
            raise RetryableError("timeout") from e

        except requests.exceptions.ConnectionError as e:
            raise RetryableError("connection error") from e

    # ---------------------------
    # 对外同步 search 方法
    # ---------------------------
    def search(
            self,
            query: str,
            count: int = 10,
            sites: Optional[list[str]] = None,
            include_raw_content: bool = False,
    ) -> dict:

        # encoded_query = quote(query)
        # 使用原始query进行查询
        params = {"q": query, "count": count, "mainText": "True"}

        if sites:
            params["sites"] = ",".join(sites)

        try:
            return self._search_with_retry(
                self.FULLTEXT_URL,
                params,
                query,
                include_raw_content
            )

        except RetryError:
            return self._empty_result(query)

        except requests.exceptions.RequestException:
            return self._empty_result(query)

    # ---------------------------
    # 响应格式转换
    # ---------------------------
    def _transform_to_tavily_format(self, data, query, request_id, include_raw_content):
        original_query = (
                query or
                (data.get("queryContext") or {}).get("originalQuery")
                or data.get("query")
        )

        result = {
            "query": original_query,
            "follow_up_questions": None,
            "answer": None,
            "images": [],
            "results": [],
            "response_time": None,
            "request_id": request_id,
        }

        web_values = []
        if isinstance(data.get("webPages"), dict) and isinstance(
                data["webPages"].get("value"), list
        ):
            web_values = data["webPages"]["value"]

        elif isinstance(data.get("results"), list):
            web_values = data["results"]

        for item in web_values:
            url = item.get("url", "")
            title = item.get("name", "")
            snippet = item.get("snippet", "")
            main_text = item.get("mainText", "")
            full_content = item.get("content", "")
            score = item.get("score", 0.0)

            if main_text:
                content_field = main_text
            elif include_raw_content:
                content_field = full_content or snippet
            else:
                content_field = snippet

            raw_content = full_content if include_raw_content else None

            result["results"].append({
                "url": url,
                "title": title,
                "content": content_field,
                "score": score,
                "raw_content": raw_content,
                "favicon": None,
            })

        return result

    def _empty_result(self, query):
        return {
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": [],
            "results": [],
            "response_time": None,
            "request_id": None,
        }


if __name__ == "__main__":
    client = CloudswaySearchClient()

    res = client.search("今天南京的天气", count=5)

    print(json.dumps(res, indent=2, ensure_ascii=False))

    client.close()
