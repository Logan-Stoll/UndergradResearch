import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import logging
import time
import random
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArticleScraper:
    def __init__(self):
        # Configure a wide range of user agents
        self.software_names = [SoftwareName.CHROME.value, SoftwareName.FIREFOX.value, 
                              SoftwareName.EDGE.value, SoftwareName.SAFARI.value]
        self.operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value, 
                                 OperatingSystem.MAC.value, OperatingSystem.ANDROID.value]
        self.user_agent_rotator = UserAgent(software_names=self.software_names, 
                                           operating_systems=self.operating_systems, limit=1000)
        
        # List of common referrers to rotate through
        self.referrers = [
            'https://www.google.com/',
            'https://www.bing.com/',
            'https://www.yahoo.com/',
            'https://duckduckgo.com/',
            'https://www.reddit.com/',
            'https://www.facebook.com/',
            'https://www.twitter.com/',
            'https://news.ycombinator.com/'
        ]
        
        self.refresh_headers()
    
    def refresh_headers(self):
        """Generate a new set of headers to avoid detection"""
        user_agent = self.user_agent_rotator.get_random_user_agent()
        referrer = random.choice(self.referrers)
        
        self.headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': referrer,
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'DNT': '1'
        }
    
    def validate_url(self, url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def get_domain(self, url):
        try:
            return urlparse(url).netloc
        except:
            return None
    
    def fetch_html(self, url):
        # Create a session to maintain cookies
        session = requests.Session()
        
        # Randomize so it doesnt look like a bot (Even though it is lol)
        time.sleep(random.uniform(1, 3))
        
        # Try multiple attempts with different strategies
        for attempt in range(1, 6):
            try:
                logger.info(f"Attempt {attempt} - Fetching {url}")
                
                self.refresh_headers()
                
                # Add randomized delay between attempts (looks more human)
                if attempt > 1:
                    delay = random.uniform(2, 5)
                    logger.info(f"Waiting {delay:.2f} seconds before retry...")
                    time.sleep(delay)
                
                # Use session to maintain cookies across requests
                response = session.get(
                    url, 
                    headers=self.headers, 
                    timeout=15,
                    allow_redirects=True
                )
                
                # Forbidden URL error 
                if response.status_code == 403:
                    logger.warning("Received 403 Forbidden - Adjusting approach for next attempt")
                    continue
                    
                # Check for success
                response.raise_for_status()
            
                return response.text
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching URL (attempt {attempt}): {e}")
            
        # All attempts failed
        logger.error(f"Failed to fetch {url} after multiple attempts")
        return None


    def extract_article_text(self, html, domain):
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove non-content elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 
                            'iframe', 'noscript', 'form', 'button', 'svg',
                            'time', 'figcaption', 'figure']):
            element.decompose()
        
        # Remove elements with common ad/non-content classes
        ad_classes = [
            'ad', 'ads', 'advertisement', 'banner', 'sponsor', 'social',
            'share', 'comment', 'newsletter', 'sidebar', 'related',
            'promotion', 'subscribe', 'subscription', 'paywall', 'premium',
            'signup', 'login', 'auth', 'modal', 'popup', 'cookie', 'consent',
            'alert', 'notification', 'message'
        ]
        
        # Find elements with these classes and remove them
        for class_name in ad_classes:
            for element in soup.find_all(class_=lambda x: x and class_name in x.lower()):
                element.decompose()
        
        # There is no good one size fits all strategy for scraping article text
        # Try each strategy and worst case scenario just return none and have the app deal with it
        
        # Strategy 1: Look for article tag
        article = soup.find('article')
        if article:
            # Find all paragraphs within the article
            paragraphs = article.find_all('p')
            if paragraphs:
                return self._clean_text(' '.join(p.get_text() for p in paragraphs))
            else:
                return self._clean_text(article.get_text())
        
        # Strategy 2: Look for common article content div classes/ids
        common_article_selectors = [
            'div.article-body', 'div.article-content', 'div.story-body',
            'div.story-content', 'div.post-content', 'div.entry-content',
            'div.content-body', 'div.article', 'div.story', 'div.content',
            'div#article-body', 'div#article-content', 'div#story-body',
            'div#story-content', 'div#post-content', 'div#content',
            'div.main-content', 'div#main-content', 'section.content',
            'div.body-text', 'div.text-content'
        ]
        
        for selector in common_article_selectors:
            parts = selector.split('.')
            tag_type = parts[0]
            
            if '#' in selector:
                parts = selector.split('#')
                tag_type = parts[0]
                identifier = parts[1]
                element = soup.find(tag_type, id=identifier)
            elif len(parts) > 1:
                identifier = parts[1]
                element = soup.find(tag_type, class_=identifier)
            else:
                element = soup.find(tag_type)
                
            if element:
                paragraphs = element.find_all('p')
                if paragraphs:
                    return self._clean_text(' '.join(p.get_text() for p in paragraphs))
                else:
                    return self._clean_text(element.get_text())
        
        # Strategy 3: Look for paragraphs within main tag
        main = soup.find('main')
        if main:
            paragraphs = main.find_all('p')
            if paragraphs:
                return self._clean_text(' '.join(p.get_text() for p in paragraphs))
        
        # Strategy 4: Look for the largest collection of paragraphs
        # This assumes the article is the largest block of text with paragraphs
        containers = soup.find_all(['div', 'section', 'article', 'main'])
        max_paragraphs = 0
        best_container = None
        
        for container in containers:
            paragraphs = container.find_all('p')
            if len(paragraphs) > max_paragraphs:
                max_paragraphs = len(paragraphs)
                best_container = container
        
        if best_container and max_paragraphs > 3:  # Require at least 3 paragraphs
            return self._clean_text(' '.join(p.get_text() for p in best_container.find_all('p')))
        
        # Strategy 5: Get all paragraphs in the document with filtering
        paragraphs = soup.find_all('p')
        if paragraphs:
            # Filter out paragraphs that are likely not part of the main content
            significant_paragraphs = []
            for p in paragraphs:
                text = p.get_text().strip()
                # Skip very short paragraphs and paragraphs that might be ads/notices
                if (len(text) > 30 and 
                    not re.search(r'(cookie|subscribe|sign up|advertisement)', text, re.IGNORECASE)):
                    significant_paragraphs.append(text)
            
            if significant_paragraphs:
                return self._clean_text(' '.join(significant_paragraphs))
        
        # Fallback: Just get text from body with basic cleaning
        body = soup.find('body')
        if body:
            return self._clean_text(body.get_text())
        
        return None
    
    def _clean_text(self, text):
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Common patterns to remove (expanded list)
        removal_patterns = [
            # Navigation and interaction elements
            r'Read more|Share|Comments|Related Articles|Subscribe|Sign up|Log in|Sign in',
            
            # Advertisement markers
            r'Advertisement|ADVERTISEMENT|Sponsored Content|Sponsored|Advert|SKIP ADVERTISEMENT',
            
            # Subscription and paywall notices
            r'You have (\d+ )?free articles? (left|remaining)|Subscribe to continue reading',
            r'You have a preview view of this article|while we are checking your access',
            r'Subscribe for( just)? \$\d+(\.\d+)?( per month| per week| per year)?',
            r'Subscribe now|Join now|Members only|Premium content',
            
            # Cookie and privacy notices
            r'We use cookies|This site uses cookies|Accept cookies|Cookie policy',
            r'Privacy policy|Terms of (use|service)',
            
            # Social media
            r'Follow us on|Like us on|Connect with us|Find us on',
            r'Twitter|Facebook|Instagram|LinkedIn|YouTube',
            
            # Newsletter signup
            r'Sign up for our newsletter|Get our newsletter|Stay updated|Subscribe to our updates',
            
            # Popups and notifications
            r'Breaking news|Latest updates|Exclusive content|Limited time offer',
            
            # Content labels
            r'Opinion|Editorial|Analysis|Featured|Special report',
            
            # Time indicators that aren't part of the actual content
            r'minutes? ago|hours? ago|days? ago|Updated [a-zA-Z]+ \d+',
            r'Published on|Published at|Posted on|Posted at',
            
            # Website identifiers
            r'Copyright © 20\d\d|All rights reserved'
        ]
        
        # Combine all patterns and remove them
        combined_pattern = '|'.join(removal_patterns)
        text = re.sub(combined_pattern, '', text, flags=re.IGNORECASE)
        
        # Remove text that looks like "X min read" or "X minute read"
        text = re.sub(r'\d+ min(ute)? read', '', text, flags=re.IGNORECASE)
        
        # Clean up any resulting double spaces
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()
    
    def extract_title(self, html):
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # It feels impossible for a one size fits all solution so here are a handful of different
        # Strategies for finding the title of the article
        
        # Idea 1: Look for article headline
        headline = soup.find(['h1', 'h2'], class_=lambda c: c and any(x in str(c).lower() for x in ['headline', 'title', 'heading']))
        if headline:
            return headline.get_text().strip()
        
        # Idea 2: Look for first h1 in the article
        article = soup.find('article')
        if article:
            h1 = article.find('h1')
            if h1:
                return h1.get_text().strip()
        
        # Strategy 3: Look for first h1 in main content
        main_content = soup.find(['main', 'div.content', 'div.article', 'div.post'])
        if main_content:
            h1 = main_content.find('h1')
            if h1:
                return h1.get_text().strip()
        
        # Strategy 4: First h1 on the page
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()
        
        # Strategy 5: Use the title tag content
        title_tag = soup.find('title')
        if title_tag:
            # Clean up title tag content which often includes site name
            title = title_tag.get_text().strip()
            # Try to remove site name which is often follows a pipe, dash, or vertical bar
            title = re.split(r'\s[-|]\s', title)[0].strip()
            return title
        
        return None

    def get_article_text(self, url):
        if not self.validate_url(url):
            return None, "Invalid URL format"

        domain = self.get_domain(url)
        
        html = self.fetch_html(url)
        if not html:
            return None, "Failed to fetch content from URL"
        
        article_title = self.extract_title(html)
        
        article_text = self.extract_article_text(html, domain)
        if not article_text or len(article_text) < 100:  # If text is too short, likely failed to extract properly
            return None, "Could not extract meaningful article content"
        
        if article_title:
            article_text = f"{article_title}\n\n{article_text}"
        
        return article_text, None

# Example usage
if __name__ == "__main__":
    scraper = ArticleScraper()
    url = "https://www.nytimes.com/2025/05/04/us/politics/joe-biden-cognitive-test-age.html"
    text, error = scraper.get_article_text(url)
    
    if error:
        print(f"Error: {error}")
    else:
        print("Article text extracted successfully:")
        print(text[:200] + "...")  # Print first 200 chars