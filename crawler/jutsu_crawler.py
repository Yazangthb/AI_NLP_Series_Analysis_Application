import scrapy
from bs4 import BeautifulSoup

class BlogSpider(scrapy.Spider):
    name = 'narutospider'
    start_urls = ['https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu']

    def parse(self, response):
        for href in response.css('.smw-columnlist-container')[0].css("a::attr(href)").extract():
            extracted_data = scrapy.Request("https://naruto.fandom.com"+href, callback=self.parse_jutsu)
            yield extracted_data

        for next_page in response.css('a.mw-nextlink'):
            yield response.follow(next_page, self.parse)
    
    def parse_jutsu(self, response):
        jutsu_name = response.css("span.mw-page-title-main::text").get()
        if jutsu_name:
            jutsu_name = jutsu_name.strip()

        div_selector = response.css("div.mw-parser-output")
        if div_selector:
            div_html = div_selector[0].get()
            soup = BeautifulSoup(div_html, 'html.parser')

            jutsu_type = ""
            aside = soup.find('aside')
            if aside:
                for cell in aside.find_all('div', {'class': 'pi-data'}):
                    if cell.find('h3'):
                        cell_name = cell.find('h3').text.strip()
                        if cell_name == "Classification":
                            jutsu_type = cell.find('div').text.strip()
                            print("========================================")
                            print(f"Jutsu Type: {jutsu_type}")
                            print("========================================")

                aside.decompose()  # remove the aside section

            jutsu_description = soup.find('p').text.strip()
            jutsu_description = jutsu_description.split('Trivia')[0].strip()

            return dict(
                jutsu_name=jutsu_name,
                jutsu_type=jutsu_type,
                jutsu_description=jutsu_description
            )

# Save the output to a JSON Lines file
if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess
    process = CrawlerProcess(settings={
        "FEEDS": {
            "./data/jutsus.jsonl": {"format": "jsonlines"},
        },
    })

    process.crawl(BlogSpider)
    process.start()
