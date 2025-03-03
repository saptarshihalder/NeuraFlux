"""
Web scraper implementation for retrieving live data.
"""

import requests
from typing import List, Dict, Optional
import re
from urllib.parse import urljoin, urlparse
import time
from bs4 import BeautifulSoup
import logging

class WebScraper:
    def __init__(self, max_depth: int = 2, delay: float = 1.0):
        self.max_depth = max_depth
        self.delay = delay
        self.visited_urls = set()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def scrape_url(self, url: str, max_pages: int = 5) -> List[Dict]:
        """
        Scrape content from a URL and its linked pages.
        
        Args:
            url: The URL to start scraping from
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of dictionaries containing scraped content
        """
        results = []
        urls_to_visit = [(url, 0)]  # (url, depth)
        
        while urls_to_visit and len(results) < max_pages:
            current_url, depth = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls or depth > self.max_depth:
                continue
            
            try:
                # Add delay to be respectful to servers
                time.sleep(self.delay)
                
                # Fetch page content
                response = requests.get(current_url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract content
                content = self._extract_content(soup)
                
                # Store result
                results.append({
                    'url': current_url,
                    'title': self._extract_title(soup),
                    'content': content,
                    'links': self._extract_links(soup, current_url)
                })
                
                # Mark as visited
                self.visited_urls.add(current_url)
                
                # Add new links to visit
                if depth < self.max_depth:
                    new_links = self._extract_links(soup, current_url)
                    urls_to_visit.extend([(link, depth + 1) for link in new_links])
                
            except Exception as e:
                self.logger.error(f"Error scraping {current_url}: {str(e)}")
                continue
        
        return results
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title = soup.title
        if title:
            return title.string.strip()
        return ""
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract and normalize links from the page."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            
            # Only include links from the same domain
            if urlparse(absolute_url).netloc == urlparse(base_url).netloc:
                links.append(absolute_url)
        
        return list(set(links))  # Remove duplicates
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Perform a web search and return results.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of dictionaries containing search results
        """
        # This is a placeholder for actual search functionality
        # In practice, you would integrate with a search API
        self.logger.warning("Web search functionality is not implemented")
        return []
    
    def extract_structured_data(self, url: str) -> Dict:
        """
        Extract structured data from a webpage.
        
        Args:
            url: URL to extract data from
            
        Returns:
            Dictionary containing structured data
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract metadata
            metadata = {
                'title': self._extract_title(soup),
                'description': self._extract_meta_description(soup),
                'keywords': self._extract_meta_keywords(soup),
                'author': self._extract_meta_author(soup),
                'date': self._extract_meta_date(soup)
            }
            
            # Extract main content sections
            content = {
                'headings': self._extract_headings(soup),
                'paragraphs': self._extract_paragraphs(soup),
                'lists': self._extract_lists(soup),
                'tables': self._extract_tables(soup)
            }
            
            return {
                'url': url,
                'metadata': metadata,
                'content': content
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting structured data from {url}: {str(e)}")
            return {}
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta description."""
        meta = soup.find('meta', attrs={'name': 'description'})
        return meta['content'] if meta else None
    
    def _extract_meta_keywords(self, soup: BeautifulSoup) -> List[str]:
        """Extract meta keywords."""
        meta = soup.find('meta', attrs={'name': 'keywords'})
        if meta and meta.get('content'):
            return [k.strip() for k in meta['content'].split(',')]
        return []
    
    def _extract_meta_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta author."""
        meta = soup.find('meta', attrs={'name': 'author'})
        return meta['content'] if meta else None
    
    def _extract_meta_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta date."""
        meta = soup.find('meta', attrs={'property': 'article:published_time'})
        return meta['content'] if meta else None
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract headings with their levels."""
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            headings.append({
                'level': int(h.name[1]),
                'text': h.get_text().strip()
            })
        return headings
    
    def _extract_paragraphs(self, soup: BeautifulSoup) -> List[str]:
        """Extract paragraphs."""
        return [p.get_text().strip() for p in soup.find_all('p')]
    
    def _extract_lists(self, soup: BeautifulSoup) -> List[List[str]]:
        """Extract lists."""
        lists = []
        for ul in soup.find_all(['ul', 'ol']):
            lists.append([li.get_text().strip() for li in ul.find_all('li')])
        return lists
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[List[List[str]]]:
        """Extract tables."""
        tables = []
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                rows.append([td.get_text().strip() for td in tr.find_all(['td', 'th'])])
            tables.append(rows)
        return tables 