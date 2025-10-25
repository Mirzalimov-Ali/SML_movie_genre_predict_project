from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from src.logger import get_logger

logger = get_logger("data_extraction", "data_extraction.log")

CHROMEDRIVER_PATH = "C:/Users/user/Downloads/chromedriver-win64/chromedriver.exe"

class DataExtraction:
    def __init__(self):
        self.chromedriver_path = CHROMEDRIVER_PATH

    def get_dataset(self, list_url, total_pages=1, genre=None):
        service = Service(self.chromedriver_path)
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(service=service, options=options)

        all_data = []
        current_id = 1

        for page in range(1, total_pages + 1):
            url = list_url if page == 1 else f"{list_url}?page={page}"
            logger.info(f"Loading page {page}: {url}")
            driver.get(url)
            time.sleep(3)  

            try:
                wait = WebDriverWait(driver, 20)
                movies = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.sc-fc35a1ef-1.lmHCrT.dli-parent")))
            except:
                logger.warning(f"No movies found on page {page}")
                continue

            for i in movies:
                # Title
                title_elements = i.find_elements(By.CSS_SELECTOR, ".ipc-title__text")
                title = title_elements[0].text.split(". ", 1)[1] if title_elements else ""

                # Metadata
                metadata = i.find_elements(By.CSS_SELECTOR, ".dli-title-metadata-item")
                year = metadata[0].text if len(metadata) > 0 else ""
                duration = metadata[1].text if len(metadata) > 1 else ""
                rated = metadata[-1].text if len(metadata) > 2 else ""

                # IMDb rating
                rating_elements = i.find_elements(By.CSS_SELECTOR, '[data-testid="ratingGroup--imdb-rating"] .ipc-rating-star--rating')
                imdb_rating = rating_elements[0].text if rating_elements else ""

                # Votes
                vote_elements = i.find_elements(By.CSS_SELECTOR, '[data-testid="ratingGroup--imdb-rating"] .ipc-rating-star--voteCount')
                votes = vote_elements[0].text if vote_elements else ""

                # Director va stars
                info_block = i.find_elements(By.CSS_SELECTOR, ".title-description-credit a")
                director = info_block[0].text if len(info_block) > 0 else ""
                stars = ", ".join([a.text for a in info_block[1:]]) if len(info_block) > 1 else ""

                # Description
                desc_elements = i.find_elements(By.CSS_SELECTOR, ".title-description-plot-container")
                desc = desc_elements[0].text if desc_elements else ""

                all_data.append({
                    "Id": current_id,
                    "Title": title,
                    "Year": year,
                    "Duration": duration,
                    "Rated": rated,
                    "IMDb_Rating": imdb_rating,
                    "Votes": votes,
                    "Director": director,
                    "Stars": stars,
                    "Genre": genre,
                    "Description": desc
                })
                current_id += 1

            logger.info(f"Page {page}: {len(movies)} movies scraped")

        driver.quit()

        df = pd.DataFrame(all_data)
        file_path = f"data/raw/{genre}_movies.csv"

        if file_path:
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
            logger.info(f"Saved {len(df)} movies to {file_path}")

        return df
