from mcp.server.fastmcp import FastMCP, Context
import httpx
from bs4 import BeautifulSoup
from typing import Any, List, Dict, Optional, Union
from dataclasses import dataclass
import urllib.parse
import sys
import traceback
import asyncio
from datetime import datetime, timedelta
import time
import re
from scholarly import scholarly
import requests

@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    position: int


class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [
            req for req in self.requests if now - req < timedelta(minutes=1)
        ]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(now)


class DuckDuckGoSearcher:
    BASE_URL = "https://html.duckduckgo.com/html"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def __init__(self):
        self.rate_limiter = RateLimiter()

    def format_results_for_llm(self, results: List[SearchResult]) -> str:
        """Format results in a natural language style that's easier for LLMs to process"""
        if not results:
            return "No results were found for your search query. This could be due to DuckDuckGo's bot detection or the query returned no matches. Please try rephrasing your search or try again in a few minutes."

        output = []
        output.append(f"Found {len(results)} search results:\n")

        for result in results:
            output.append(f"{result.position}. {result.title}")
            output.append(f"   URL: {result.link}")
            output.append(f"   Summary: {result.snippet}")
            output.append("")  # Empty line between results

        return "\n".join(output)

    async def search(
        self, query: str, ctx: Context, max_results: int = 10
    ) -> List[SearchResult]:
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()

            # Create form data for POST request
            data = {
                "q": query,
                "b": "",
                "kl": "",
            }

            await ctx.info(f"Searching DuckDuckGo for: {query}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL, data=data, headers=self.HEADERS, timeout=30.0
                )
                response.raise_for_status()

            # Parse HTML response
            soup = BeautifulSoup(response.text, "html.parser")
            if not soup:
                await ctx.error("Failed to parse HTML response")
                return []

            results = []
            for result in soup.select(".result"):
                title_elem = result.select_one(".result__title")
                if not title_elem:
                    continue

                link_elem = title_elem.find("a")
                if not link_elem:
                    continue

                title = link_elem.get_text(strip=True)
                link = link_elem.get("href", "")

                # Skip ad results
                if "y.js" in link:
                    continue

                # Clean up DuckDuckGo redirect URLs
                if link.startswith("//duckduckgo.com/l/?uddg="):
                    link = urllib.parse.unquote(link.split("uddg=")[1].split("&")[0])

                snippet_elem = result.select_one(".result__snippet")
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                results.append(
                    SearchResult(
                        title=title,
                        link=link,
                        snippet=snippet,
                        position=len(results) + 1,
                    )
                )

                if len(results) >= max_results:
                    break

            await ctx.info(f"Successfully found {len(results)} results")
            return results

        except httpx.TimeoutException:
            await ctx.error("Search request timed out")
            return []
        except httpx.HTTPError as e:
            await ctx.error(f"HTTP error occurred: {str(e)}")
            return []
        except Exception as e:
            await ctx.error(f"Unexpected error during search: {str(e)}")
            traceback.print_exc(file=sys.stderr)
            return []


class WebContentFetcher:
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=20)

    async def fetch_and_parse(self, url: str, ctx: Context) -> str:
        """Fetch and parse content from a webpage"""
        try:
            await self.rate_limiter.acquire()

            await ctx.info(f"Fetching content from: {url}")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                    follow_redirects=True,
                    timeout=30.0,
                )
                response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Get the text content
            text = soup.get_text()

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000] + "... [content truncated]"

            await ctx.info(
                f"Successfully fetched and parsed content ({len(text)} characters)"
            )
            return text

        except httpx.TimeoutException:
            await ctx.error(f"Request timed out for URL: {url}")
            return "Error: The request timed out while trying to fetch the webpage."
        except httpx.HTTPError as e:
            await ctx.error(f"HTTP error occurred while fetching {url}: {str(e)}")
            return f"Error: Could not access the webpage ({str(e)})"
        except Exception as e:
            await ctx.error(f"Error fetching content from {url}: {str(e)}")
            return f"Error: An unexpected error occurred while fetching the webpage ({str(e)})"


# Initialize FastMCP server
mcp = FastMCP("ddg-search")
searcher = DuckDuckGoSearcher()
fetcher = WebContentFetcher()
def google_scholar_search(query, num_results=5):
    """
    Function to search Google Scholar using a simple keyword query.
    
    Parameters:
    query (str): The search query (e.g., paper title or author).
    num_results (int): The number of results to retrieve.
    
    Returns:
    list: A list of dictionaries containing search results.
    """
    # Prepare the search URL
    search_url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}"
    
    # Set up headers to mimic a real browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Send the GET request to Google Scholar
    response = requests.get(search_url, headers=headers)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to fetch data. HTTP Status code: {response.status_code}")
        return []

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all the articles in the search results
    results = []
    count = 0

    # Find the results on the page
    for item in soup.find_all('div', class_='gs_ri'):
        if count >= num_results:
            break

        title_tag = item.find('h3', class_='gs_rt')
        title = title_tag.get_text() if title_tag else 'No title available'

        link = title_tag.find('a')['href'] if title_tag and title_tag.find('a') else 'No link available'

        authors_tag = item.find('div', class_='gs_a')
        authors = authors_tag.get_text() if authors_tag else 'No authors available'

        abstract_tag = item.find('div', class_='gs_rs')
        abstract = abstract_tag.get_text() if abstract_tag else 'No abstract available'

        result_data = {
            'Title': title,
            'Authors': authors,
            'Abstract': abstract,
            'URL': link
        }
        results.append(result_data)
        count += 1

    return results

def advanced_google_scholar_search(query, author=None, year_range=None, num_results=5):
    """
    Function to search Google Scholar using advanced search filters (e.g., author, year range).
    
    Parameters:
    query (str): The search query (e.g., paper title or topic).
    author (str): The author's name to filter the results (default is None).
    year_range (tuple): A tuple (start_year, end_year) to filter the results by publication year (default is None).
    num_results (int): The number of results to retrieve.
    
    Returns:
    list: A list of dictionaries containing search results.
    """
    # Prepare the advanced search URL
    search_url = "https://scholar.google.com/scholar?"
    
    # Build the search query
    search_params = {'q': query.replace(' ', '+')}
    if author:
        search_params['as_auth'] = author
    if year_range:
        start_year, end_year = year_range
        search_params['as_ylo'] = start_year  # Start year
        search_params['as_yhi'] = end_year  # End year
    
    # Encode the search parameters into the URL
    search_url += '&'.join([f"{key}={value}" for key, value in search_params.items()])

    # Set up headers to mimic a real browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Send the GET request to Google Scholar
    response = requests.get(search_url, headers=headers)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to fetch data. HTTP Status code: {response.status_code}")
        return []

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all the articles in the search results
    results = []
    count = 0

    # Find the results on the page
    for item in soup.find_all('div', class_='gs_ri'):
        if count >= num_results:
            break

        title_tag = item.find('h3', class_='gs_rt')
        title = title_tag.get_text() if title_tag else 'No title available'

        link = title_tag.find('a')['href'] if title_tag and title_tag.find('a') else 'No link available'

        authors_tag = item.find('div', class_='gs_a')
        authors = authors_tag.get_text() if authors_tag else 'No authors available'

        abstract_tag = item.find('div', class_='gs_rs')
        abstract = abstract_tag.get_text() if abstract_tag else 'No abstract available'

        result_data = {
            'Title': title,
            'Authors': authors,
            'Abstract': abstract,
            'URL': link
        }
        results.append(result_data)
        count += 1

    return results
@mcp.tool()
async def search_google_scholar_key_words(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for articles on Google Scholar using key words.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 5)

    Returns:
        List of dictionaries containing article information
    """
    try:
        results = await asyncio.to_thread(google_scholar_search, query, num_results)
        return results
    except Exception as e:
        return [{"error": f"An error occurred while searching Google Scholar: {str(e)}"}]

@mcp.tool()
async def search_google_scholar_advanced(query: str, author: Optional[str] = None, year_range: Optional[tuple] = None, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for articles on Google Scholar using advanced filters.

    Args:
        query: General search query
        author: Author name
        year_range: tuple containing (start_year, end_year)
        num_results: Number of results to return (default: 5)

    Returns:
        List of dictionaries containing article information
    """
    try:
        results = await asyncio.to_thread(
            advanced_google_scholar_search,
            query, author, year_range, num_results
        )
        return results
    except Exception as e:
        return [{"error": f"An error occurred while performing advanced search on Google Scholar: {str(e)}"}]

@mcp.tool()
async def get_author_info(author_name: str) -> Dict[str, Any]:
    """
    Get detailed information about an author from Google Scholar.

    Args:
        author_name: Name of the author to search for

    Returns:
        Dictionary containing author information
    """
    try:
        search_query = scholarly.search_author(author_name)
        author = await asyncio.to_thread(next, search_query)
        filled_author = await asyncio.to_thread(scholarly.fill, author)
        
        # Extract relevant information
        author_info = {
            "name": filled_author.get("name", "N/A"),
            "affiliation": filled_author.get("affiliation", "N/A"),
            "interests": filled_author.get("interests", []),
            "citedby": filled_author.get("citedby", 0),
            "publications": [
                {
                    "title": pub.get("bib", {}).get("title", "N/A"),
                    "year": pub.get("bib", {}).get("pub_year", "N/A"),
                    "citations": pub.get("num_citations", 0)
                }
                for pub in filled_author.get("publications", [])[:5]  # Limit to top 5 publications
            ]
        }
        return author_info
    except Exception as e:
        return {"error": f"An error occurred while retrieving author information: {str(e)}"}

@mcp.tool()
async def search(query: str, ctx: Context, max_results: int = 20) -> str:
    """
    Search DuckDuckGo and return formatted results.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 10)
        ctx: MCP context for logging
    """
    try:
        results = await searcher.search(query, ctx, max_results)
        return searcher.format_results_for_llm(results)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return f"An error occurred while searching: {str(e)}"


@mcp.tool()
async def fetch_content(url: str, ctx: Context) -> str:
    """
    Fetch and parse content from a webpage URL.

    Args:
        url: The webpage URL to fetch content from
        ctx: MCP context for logging
    """
    return await fetcher.fetch_and_parse(url, ctx)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
