import scrapy


class FraunhoferInstitutesSpider(scrapy.Spider):
    name = 'fraunhofer_institutes'
    start_urls = ['https://www.fraunhofer.de/en/institutes/institutes-and-research-establishments-in-germany.html']

    def parse(self, response):
        # Extracting all rows from the table
        rows = response.css('.table-content-row')

        for row in rows:
            # Extracting institute name and link from each row
            institute_name = row.css('.table-cell-name a::text').get()
            institute_link = row.css('.table-cell-name a::attr(href)').get()

            # Yielding the item with institute name and link
            yield {
                'institute': institute_name,
                'link': institute_link
            }
