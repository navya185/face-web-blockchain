import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from app.search.models import SearchResult

load_dotenv()

class WebSearchEngine:
    """
    Real Web & Social Media Search Integration Engine.
    Executes actual web queries via SerpAPI, Google CSE, Bing API, or live DDGS / HTML page extractors.
    """
    def __init__(self):
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        self.bing_api_key = os.getenv("BING_SEARCH_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    def extract_domain(self, url: str) -> str:
        """Extracts domain/platform name from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain or "web"
        except Exception:
            return "web"

    def search_serpapi(self, query: str, image_url: Optional[str] = None, max_results: int = 5) -> List[SearchResult]:
        """Queries SerpAPI Google Lens / Reverse Image / Google Search."""
        if not self.serpapi_key:
            return []

        results: List[SearchResult] = []
        try:
            if image_url:
                endpoint = "https://serpapi.com/search.json"
                params = {
                    "engine": "google_lens",
                    "url": image_url,
                    "api_key": self.serpapi_key
                }
                resp = self.session.get(endpoint, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    visual_matches = data.get("visual_matches", [])
                    for idx, match in enumerate(visual_matches[:max_results]):
                        res_url = match.get("link", "")
                        results.append(SearchResult(
                            url=res_url,
                            platform=self.extract_domain(res_url),
                            title=match.get("title", f"Visual Match {idx+1}"),
                            timestamp=time.strftime("%Y-%m-%d"),
                            author=match.get("source", "Web"),
                            image_url=match.get("thumbnail"),
                            raw_content=match.get("title", ""),
                            relevance_score=round(1.0 - (idx * 0.05), 2),
                            extra_metadata={"engine": "serpapi_google_lens"}
                        ))
            else:
                endpoint = "https://serpapi.com/search.json"
                params = {
                    "engine": "google",
                    "q": query,
                    "api_key": self.serpapi_key
                }
                resp = self.session.get(endpoint, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    organic = data.get("organic_results", [])
                    for idx, item in enumerate(organic[:max_results]):
                        res_url = item.get("link", "")
                        results.append(SearchResult(
                            url=res_url,
                            platform=self.extract_domain(res_url),
                            title=item.get("title", "Search Result"),
                            timestamp=item.get("date", time.strftime("%Y-%m-%d")),
                            author=self.extract_domain(res_url),
                            image_url=item.get("thumbnail"),
                            raw_content=item.get("snippet", ""),
                            relevance_score=round(0.95 - (idx * 0.05), 2),
                            extra_metadata={"engine": "serpapi_google"}
                        ))
        except Exception as e:
            print(f"[WebSearchEngine] SerpAPI warning: {e}")

        return results

    def search_google_cse(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Queries Google Custom Search Engine (CSE) API."""
        if not (self.google_api_key and self.google_cse_id):
            return []

        results: List[SearchResult] = []
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "num": min(max_results, 10)
            }
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])
                for idx, item in enumerate(items[:max_results]):
                    res_url = item.get("link", "")
                    pagemap = item.get("pagemap", {})
                    cse_image = None
                    if "cse_image" in pagemap and pagemap["cse_image"]:
                        cse_image = pagemap["cse_image"][0].get("src")
                    
                    results.append(SearchResult(
                        url=res_url,
                        platform=self.extract_domain(res_url),
                        title=item.get("title", "Search Result"),
                        timestamp=time.strftime("%Y-%m-%d"),
                        author=self.extract_domain(res_url),
                        image_url=cse_image,
                        raw_content=item.get("snippet", ""),
                        relevance_score=round(0.90 - (idx * 0.05), 2),
                        extra_metadata={"engine": "google_cse"}
                    ))
        except Exception as e:
            print(f"[WebSearchEngine] Google CSE warning: {e}")

        return results

    def search_ddgs(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Executes live DDGS open web search."""
        results: List[SearchResult] = []
        try:
            DDGS_class = None
            try:
                from ddgs import DDGS
                DDGS_class = DDGS
            except ImportError:
                from duckduckgo_search import DDGS
                DDGS_class = DDGS

            ddgs_instance = DDGS_class()
            raw_results = list(ddgs_instance.text(query, max_results=max_results))

            for idx, item in enumerate(raw_results):
                res_url = item.get("href", "")
                title = item.get("title", "")
                snippet = item.get("body", "")

                results.append(SearchResult(
                    url=res_url,
                    platform=self.extract_domain(res_url),
                    title=title,
                    timestamp=time.strftime("%Y-%m-%d"),
                    author=self.extract_domain(res_url),
                    image_url=None,
                    raw_content=f"{title}\n{snippet}",
                    relevance_score=round(0.88 - (idx * 0.04), 2),
                    extra_metadata={"engine": "ddgs"}
                ))
        except Exception as e:
            print(f"[WebSearchEngine] DDGS search error: {e}")

        return results

    def fetch_web_page_metadata(self, url: str) -> Optional[SearchResult]:
        """Fetches a web page URL directly and extracts title, images, and content."""
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else "Discovered Web Content"

            author = "Unknown"
            meta_author = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", property="article:author")
            if meta_author and meta_author.get("content"):
                author = meta_author["content"].strip()
            else:
                author = self.extract_domain(url)

            date_str = time.strftime("%Y-%m-%d")
            meta_date = soup.find("meta", property="article:published_time") or soup.find("meta", attrs={"name": "date"})
            if meta_date and meta_date.get("content"):
                date_str = meta_date["content"][:10]

            image_url = None
            og_img = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "twitter:image"})
            if og_img and og_img.get("content"):
                image_url = og_img["content"]
            else:
                imgs = soup.find_all("img")
                for img in imgs:
                    src = img.get("src")
                    if src and (src.endswith(".jpg") or src.endswith(".png") or src.endswith(".jpeg")):
                        if src.startswith("http"):
                            image_url = src
                            break

            for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
                script_or_style.decompose()
            text_content = soup.get_text(separator=" ", strip=True)[:2000]

            return SearchResult(
                url=url,
                platform=self.extract_domain(url),
                title=title,
                timestamp=date_str,
                author=author,
                image_url=image_url,
                raw_content=f"{title}\n{text_content}",
                relevance_score=0.95,
                extra_metadata={"engine": "direct_scrape"}
            )
        except Exception as e:
            print(f"[WebSearchEngine] Error scraping target URL {url}: {e}")
            return None

    def search(
        self,
        query: str = "person profile face image",
        image_url: Optional[str] = None,
        target_url: Optional[str] = None,
        max_results: int = 5
    ) -> List[SearchResult]:
        """Executes search across configured backends in priority order."""
        results: List[SearchResult] = []

        # 1. Direct URL inspection if supplied
        if target_url:
            direct_res = self.fetch_web_page_metadata(target_url)
            if direct_res:
                results.append(direct_res)

        # 2. SerpAPI
        if self.serpapi_key:
            results.extend(self.search_serpapi(query=query, image_url=image_url, max_results=max_results))

        # 3. Google CSE
        if self.google_api_key and self.google_cse_id and len(results) < max_results:
            results.extend(self.search_google_cse(query=query, max_results=max_results - len(results)))

        # 4. DDGS Search
        if len(results) < max_results:
            results.extend(self.search_ddgs(query=query, max_results=max_results - len(results)))

        return results[:max_results]
