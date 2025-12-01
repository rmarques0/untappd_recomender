import csv
import time
import logging
import os
import sys
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

# Agregar el directorio raíz al path para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.helpers import create_selenium_driver, login_untappd, click_show_more

class BeerScraper:
    """Scraper para cervejas de cervecerías argentinas"""
    
    def __init__(self, login_driver=None, use_headless=False):
        self.base_url = "https://untappd.com"
        self.breweries_file = "data_collection/data/breweries.csv"
        self.output_file = "data_collection/data/beers.csv"
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configurar Selenium con optimización de memoria
        if login_driver and use_headless:
            # Usar cookies del login driver y crear nuevo driver headless
            from utils.helpers import create_logged_driver
            self.driver = create_logged_driver(headless=True, transfer_from=login_driver)
            self.logger.info("BeerScraper listo con sesión reutilizada (headless)")
        else:
            # Comportamiento original para compatibilidad
            self.driver = create_selenium_driver(headless=False)
            if not login_untappd(self.driver):
                raise Exception("No se pudo hacer login")
            self.logger.info("BeerScraper listo con login propio")
    
    def collect_all_beers(self, max_breweries=None, max_clicks_per_brewery=5):
        """Recolecta cervejas de todas las cervecerías"""
        # Verificar si debe saltar este paso
        if max_breweries == 0:
            self.logger.info("SALTANDO recolección de cervejas (max_breweries=0)")
            return []
        
        # Leer cervecerías del CSV
        breweries = self.load_breweries()
        if max_breweries:
            breweries = breweries[:max_breweries]
        
        self.logger.info(f"Iniciando extracción de cervejas de {len(breweries)} cervecerías...")
        
        all_beers = []
        
        for i, brewery in enumerate(breweries, 1):
            try:
                self.logger.info(f"[{i}/{len(breweries)}] {brewery['name']}")
                
                # Recolectar cervezas de esta cervecería
                brewery_beers = self.collect_brewery_beers(brewery, max_clicks_per_brewery)
                all_beers.extend(brewery_beers)
                
                # Delay entre cervecerías
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error en {brewery['name']}: {e}")
                continue
        
        self.logger.info(f"Extracción finalizada: {len(all_beers)} cervejas encontradas")
        return all_beers
    
    def load_breweries(self):
        """Cargar cervecerías del CSV"""
        breweries = []
        with open(self.breweries_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                breweries.append(row)
        return breweries
    
    def collect_brewery_beers(self, brewery, max_clicks=5):
        """Recolecta cervejas de uma cervecería específica"""
        brewery_url = brewery['url']
        brewery_id = brewery['brewery_id']
        brewery_name = brewery['name']
        
        try:
            # Ir a la página de cervejas de la cervecería (URL/beer)
            beer_page_url = f"{brewery_url}/beer"
            self.logger.info(f"Navegando a: {beer_page_url}")
            
            self.driver.get(beer_page_url)
            time.sleep(3)
            
            # Hacer clic en "Show More" para cargar más cervejas
            click_show_more(self.driver, max_clicks, page_type="brewery")
            
            # Extrair cervejas de la página
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            beer_elements = soup.find_all('div', class_='beer-item')
            
            beers = []
            for beer_element in beer_elements:
                beer_data = self.extract_beer_data(beer_element, brewery_id, brewery_name)
                if beer_data:
                    beers.append(beer_data)
            
            return beers
            
        except Exception as e:
            self.logger.error(f"Error en cervejas de {brewery_name}: {e}")
            return []
    
    def extract_beer_data(self, beer_element, brewery_id, brewery_name):
        """Extrae datos de uma cerveja da página /beer"""
        try:
            # Nombre de la cerveja (h4 com link)
            name_elem = beer_element.find('h4')
            if not name_elem:
                # Fallback para estrutura alternativa
                name_elem = beer_element.find('p', class_='name')
            
            if not name_elem:
                return None
                
            name_link = name_elem.find('a')
            if not name_link:
                return None
                
            beer_name = name_link.get_text(strip=True)
            beer_url = urljoin(self.base_url, name_link['href'])
            
            # Extraer beer_id de la URL
            beer_id = self.extract_beer_id(beer_url)
            
            # URL de la imagen de la cerveja
            image_url = ''
            img_elem = beer_element.find('img')
            if img_elem and img_elem.get('src'):
                image_url = urljoin(self.base_url, img_elem['src'])
            
            # Estilo de la cerveja 
            style_elem = beer_element.find('p', class_='style') or beer_element.find('.style')
            beer_style = style_elem.get_text(strip=True) if style_elem else ''
            
            # ABV 
            abv = '0'
            abv_elem = beer_element.find('p', class_='abv')
            if abv_elem:
                abv_text = abv_elem.get_text(strip=True)
                abv = self.extract_number(abv_text, is_float=True)
            
            # IBU
            ibu = '0'
            ibu_elem = beer_element.find('p', class_='ibu')
            if ibu_elem:
                ibu_text = ibu_elem.get_text(strip=True)
                ibu = self.extract_number(ibu_text)
            
            # Rating (pode estar em div.caps ou span)
            rating = '0'
            rating_elem = beer_element.find('div', class_='caps')
            if rating_elem:
                rating = rating_elem.get('data-rating', '0')
            else:
                # Fallback: buscar por patrón (X.XX)
                rating_text = beer_element.get_text()
                rating_match = re.search(r'\((\d+\.?\d*)\)', rating_text)
                if rating_match:
                    rating = rating_match.group(1)
            
            # Total de ratings 
            total_ratings = '0'
            # Buscar por patrón "X,XXX Ratings"
            text_content = beer_element.get_text()
            ratings_match = re.search(r'([\d,]+)\s+Ratings?', text_content)
            if ratings_match:
                total_ratings = ratings_match.group(1).replace(',', '')
            
            return {
                'beer_id': beer_id,
                'beer_name': beer_name,
                'brewery_id': brewery_id,
                'brewery_name': brewery_name,
                'style': beer_style,
                'abv': abv,
                'ibu': ibu,
                'rating': rating,
                'total_ratings': total_ratings,
                'url': beer_url,
                'image_url': image_url
            }
            
        except Exception as e:
            self.logger.error(f"Error extrayendo datos de cerveja: {e}")
            return None
    
    def extract_beer_id(self, url):
        """Extrae el ID de la cerveja de la URL"""
        try:
            # URL format: https://untappd.com/b/beer-name/12345
            match = re.search(r'/b/.+/(\d+)', url)
            if match:
                return match.group(1)
            return ''
        except:
            return ''
    
    def extract_number(self, text, is_float=False):
        """Extrae números de um texto"""
        if not text:
            return '0'
        
        # Para números con decimales (ABV)
        if is_float:
            # Buscar patrón como "5.2%" o "5.2"
            match = re.search(r'(\d+\.?\d*)', text)
            if match:
                return match.group(1)
            return '0'
        
        # Para números enteros (IBU, ratings)
        numbers = re.findall(r'[\d,]+', text)
        if numbers:
            return numbers[0].replace(',', '')
        return '0'
    
    def save_to_csv(self, beers):
        """Guarda las cervejas en CSV"""
        if not beers:
            self.logger.warning("No hay datos para guardar")
            return
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'beer_id', 'beer_name', 'brewery_id', 'brewery_name', 
                'style', 'abv', 'ibu', 'rating', 'total_ratings', 'url', 'image_url'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for beer in beers:
                writer.writerow(beer)
        
        self.logger.info(f"Datos guardados en {self.output_file}")
    
    def close(self):
        """Cierra el driver de Selenium"""
        if self.driver:
            self.driver.quit()
            self.logger.info("Driver cerrado")

def main():
    """Función principal"""
    scraper = BeerScraper()
    
    try:
        # Recolectar cervezas (limitar a 2 cervecerías para prueba inicial)
        beers = scraper.collect_all_beers(max_breweries=2, max_clicks_per_brewery=3)
        
        # Guardar en CSV
        scraper.save_to_csv(beers)
        
    finally:
        # Cerrar driver
        scraper.close()

if __name__ == "__main__":
    main()