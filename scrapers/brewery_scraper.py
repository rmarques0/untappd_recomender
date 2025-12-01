import csv
import time
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import sys
import os

# Agregar el directorio raíz al path para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.helpers import create_selenium_driver, login_untappd, click_show_more

class BreweryScraper:
    """Scraper para cervecerías argentinas"""
    
    def __init__(self, login_driver=None, use_headless=False):
        self.base_url = "https://untappd.com"
        self.output_file = "data_collection/data/breweries.csv"
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configurar Selenium con optimización de memoria
        if login_driver and use_headless:
            # Usar cookies del login driver y crear nuevo driver headless
            from utils.helpers import create_logged_driver
            self.driver = create_logged_driver(headless=True, transfer_from=login_driver)
            self.logger.info("BreweryScraper listo con sesión reutilizada (headless)")
        else:
            # Comportamiento original para compatibilidad
            self.driver = create_selenium_driver(headless=False)
            if not login_untappd(self.driver):
                raise Exception("No se pudo hacer login")
            self.logger.info("BreweryScraper listo con login propio")
    
    def collect_breweries(self, max_clicks=10):
        """Recolecta cervecerías"""
        # Verificar si debe saltar este paso
        if max_clicks == 0:
            self.logger.info("SALTANDO recolección de cervecerías (max_clicks=0)")
            return []
        
        breweries = []
        
        self.logger.info("Iniciando extracción de cervecerías...")
        
        try:
            # Ir a la página de búsqueda
            self.driver.get("https://untappd.com/search?q=argentina&type=brewery")
            time.sleep(5)
            
            # Hacer clic en "Show More" para cargar más resultados
            click_show_more(self.driver, max_clicks)
            
            # Extraer todas las cervecerías
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            brewery_elements = soup.find_all('div', class_='beer-item')
            
            for brewery_element in brewery_elements:
                brewery_data = self.extract_brewery_data(brewery_element)
                if brewery_data:
                    # Verificar duplicados
                    if not any(b['brewery_id'] == brewery_data['brewery_id'] for b in breweries):
                        breweries.append(brewery_data)
                        
        except Exception as e:
            self.logger.error(f"Error durante extracción: {e}")
        
        self.logger.info(f"Extracción finalizada: {len(breweries)} cervecerías encontradas")
        return breweries
    
    def extract_brewery_data(self, brewery_element):
        """
        Extrae datos de un elemento de brewery
        """
        try:
            # Nombre de la brewery
            name_elem = brewery_element.find('p', class_='name').find('a')
            name = name_elem.get_text(strip=True) if name_elem else ''

            # Ubicación (primer p.style)
            style_elements = brewery_element.find_all('p', class_='style')
            location = style_elements[0].get_text(strip=True) if len(style_elements) > 0 else ''

            # Tipo de brewery (segundo p.style)
            brewery_type = style_elements[1].get_text(strip=True) if len(style_elements) > 1 else ''

            # Total de cervejas
            beers_elem = brewery_element.find('p', class_='abv')
            beer_count_text = beers_elem.get_text(strip=True) if beers_elem else ''
            beer_count = self.extract_number(beer_count_text)

            # Total de ratings
            ratings_elem = brewery_element.find('p', class_='ibu')
            total_ratings_text = ratings_elem.get_text(strip=True) if ratings_elem else ''
            total_ratings = self.extract_number(total_ratings_text)

            # Rating promedio
            rating_elem = brewery_element.find('div', class_='caps')
            rating = rating_elem.get('data-rating', '0') if rating_elem else '0'

            # URL de la brewery
            url = urljoin(self.base_url, name_elem['href']) if name_elem else ''

            # Extraer brewery_id de la URL
            brewery_id = self.extract_brewery_id(url)

            return {
                'brewery_id': brewery_id,
                'name': name,
                'location': location,
                'type': brewery_type,
                'rating': rating,
                'total_ratings': total_ratings,
                'beer_count': beer_count,
                'url': url
            }

        except Exception as e:
            self.logger.error(f"Error al extraer datos de brewery: {e}")
            return None

    def extract_brewery_id(self, url):
        """
        Extrae el ID de la brewery de la URL
        """
        try:
            # Ejemplo: https://untappd.com/w/cerveceria-y-malteria-quilmes/5415
            parts = url.split('/')
            if len(parts) >= 2:
                return parts[-1]  # Último elemento
            return url.split('/')[-1] if url else ''
        except:
            return ''

    def extract_number(self, text):
        """
        Extrae solo números de un texto
        Ejemplo: "107,481 Ratings" -> "107481"
        """
        import re
        if not text:
            return '0'

        # Buscar números (incluyendo comas)
        numbers = re.findall(r'[\d,]+', text)
        if numbers:
            # Tomar el primer número y remover comas
            return numbers[0].replace(',', '')
        return '0'

    def save_to_csv(self, breweries):
        """
        Guarda las cervecerías en CSV
        """
        if not breweries:
            self.logger.warning("No hay datos para guardar")
            return
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['brewery_id', 'name', 'location', 'type', 'rating', 'total_ratings', 'beer_count', 'url']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for brewery in breweries:
                writer.writerow(brewery)
        
        self.logger.info(f"Datos guardados en {self.output_file}")
    
    def close(self):
        """Cierra el driver de Selenium"""
        if self.driver:
            self.driver.quit()
            self.logger.info("Driver cerrado")

def main():
    """
    Función principal
    """
    scraper = BreweryScraper()
    
    try:
        # Recolectar cervecerías
        breweries = scraper.collect_breweries()
        
        # Guardar en CSV
        scraper.save_to_csv(breweries)
        
    finally:
        # Cerrar driver
        scraper.close()

if __name__ == "__main__":
    main()
