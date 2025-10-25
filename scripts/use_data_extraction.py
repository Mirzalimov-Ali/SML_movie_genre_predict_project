from src.data_extraction import DataExtraction
from src.logger import get_logger

logger = get_logger("use_data_extraction", "data_extraction.log")

# ======================================= Movies List URLS =======================================
HORROR_LIST_URL = "https://www.imdb.com/list/ls021161997/"          
ACTION_LIST_URL = "https://www.imdb.com/list/ls070233852/"  
FAMILY_LIST_URL = "https://www.imdb.com/es-es/list/ls096388064/"  
SPORT_LIST_URL = "https://www.imdb.com/list/ls054945956/"  


# ======================================= Horror Movies =======================================
try:
    extractor = DataExtraction()
    extractor.get_dataset(list_url=HORROR_LIST_URL, total_pages=4, genre="Horror")
    logger.info("Horror movies successfully scraped!")
except Exception as e:
    logger.error(f"Horror movies scraping failed: {e}")


# ======================================= Action Movies =======================================
try:
    extractor = DataExtraction()
    extractor.get_dataset(list_url=ACTION_LIST_URL, total_pages=4, genre="Action")
    logger.info("Action movies successfully scraped!")
except Exception as e:
    logger.error(f"Action movies scraping failed: {e}")


# ======================================= Comedy Movies =======================================
try:
    extractor = DataExtraction()
    extractor.get_dataset(list_url=FAMILY_LIST_URL, total_pages=2, genre="Family")
    logger.info("Family movies successfully scraped!")
except Exception as e:
    logger.error(f"Family movies scraping failed: {e}")


# ======================================= Comedy Movies =======================================
try:
    extractor = DataExtraction()
    extractor.get_dataset(list_url=SPORT_LIST_URL, total_pages=6, genre="Sport")
    logger.info("Sport movies successfully scraped!")
except Exception as e:
    logger.error(f"Sport movies scraping failed: {e}")
