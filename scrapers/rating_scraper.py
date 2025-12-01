import csv
import time
import logging
import os
import sys
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

# Agregar el directorio raíz al path para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.helpers import create_selenium_driver, login_untappd, click_show_more

class RatingScraper:
    """Scraper para ratings de cervejas argentinas"""
    
    def __init__(self, login_driver=None, use_headless=False, batch_size=50):
        self.base_url = "https://untappd.com"
        self.beers_file = "data_collection/data/beers.csv"
        self.output_file = "data_collection/data/beer_ratings.csv"
        self.checkpoint_file = "data_collection/data/rating_checkpoint.json"
        self.batch_size = batch_size
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configurar Selenium con optimización de memoria
        if login_driver and use_headless:
            # Usar cookies del login driver y crear nuevo driver headless
            from utils.helpers import create_logged_driver
            self.driver = create_logged_driver(headless=True, transfer_from=login_driver)
            self.logger.info("RatingScraper listo con sesión reutilizada (headless)")
        else:
            # Comportamiento original para compatibilidad
            self.driver = create_selenium_driver(headless=False)
            if not login_untappd(self.driver):
                raise Exception("No se pudo hacer login")
            self.logger.info("RatingScraper listo con login propio")
    
    def collect_all_ratings(self, max_beers=None, min_ratings=100, max_clicks_per_beer=5):
        """Recolecta ratings de cervejas populares con salvamento en batches"""
        # Verificar si debe saltar este paso
        if max_beers == 0:
            self.logger.info("SALTANDO recolección de ratings (max_beers=0)")
            return []
        
        # Leer cervejas del CSV
        beers = self.load_beers()
        
        # Filtrar cervejas por popularidade
        original_beer_count = len(beers)
        popular_beers = [b for b in beers if int(b['total_ratings']) >= min_ratings]
        filtered_count = original_beer_count - len(popular_beers)
        
        self.logger.info(f"Filtro de popularidad: {len(popular_beers)}/{original_beer_count} cervejas con ≥{min_ratings} ratings (filtradas: {filtered_count})")
        
        # Ordenar por popularidade (mais ratings primeiro)
        popular_beers.sort(key=lambda x: int(x['total_ratings']), reverse=True)
        
        # Obtener cervejas ya procesadas
        already_scraped_beers = self.get_already_scraped_beers()
        total_ratings_count = self.get_total_ratings_count()
        
        self.logger.info(f"Estado actual: {len(already_scraped_beers)} cervejas ya procesadas con {total_ratings_count} ratings totales")
        
        # Encontrar el próximo índice a procesar
        next_index = self.find_next_beer_to_process(popular_beers, already_scraped_beers)
        
        if next_index >= len(popular_beers):
            self.logger.info("¡Todas las cervejas ya han sido procesadas!")
            return []
        
        # Continuar desde el próximo índice
        remaining_beers = popular_beers[next_index:]
        
        if max_beers:
            remaining_beers = remaining_beers[:max_beers]
        
        self.logger.info(f"Continuando desde índice {next_index}: {len(remaining_beers)} cervejas restantes por procesar")
        self.logger.info(f"Salvamento en batches de {self.batch_size} cervejas")
        
        all_ratings = []
        batch_ratings = []
        successful_beers = 0  # Contador de cervejas processadas com sucesso
        current_processed_count = len(already_scraped_beers)  # Mantener contador en memoria
        
        for i, beer in enumerate(remaining_beers, 1):
            try:
                # Encontrar el índice original de esta cerveja en la lista completa
                original_beer_index = next(idx for idx, b in enumerate(beers) if b['beer_id'] == beer['beer_id'])
                current_total_ratings = self.get_total_ratings_count()
                self.logger.info(f"[{i}/{len(remaining_beers)}] {beer['beer_name']} - {beer['brewery_name']} (índice original: {original_beer_index})")
                
                # Recolectar ratings desta cerveja
                beer_ratings = self.collect_beer_ratings(beer, max_clicks_per_beer)
                
                # Só conta como sucesso se conseguiu ratings
                if beer_ratings:
                    batch_ratings.extend(beer_ratings)
                    all_ratings.extend(beer_ratings)
                    successful_beers += 1
                    current_processed_count += 1  # Incrementar contador en memoria
                    new_total = self.get_total_ratings_count()
                    self.logger.info(f"✅ Sucesso: {len(beer_ratings)} ratings coletados")
                    # Log de progresso total (só quando há sucesso)
                    self.logger.info(f"Total acumulado: {new_total} ratings totales de {current_processed_count} cervejas procesadas")
                else:
                    self.logger.warning(f"⚠️ Nenhum rating coletado para {beer['beer_name']}")
                
                # Salvar em batch (só se há ratings para salvar)
                if batch_ratings and (i % self.batch_size == 0 or i == len(remaining_beers)):
                    self.save_batch_to_csv(batch_ratings, append=True)  # Siempre append ya que ya hay datos
                    if self.batch_size == 1:
                        self.logger.info(f"Cerveja guardada: {len(batch_ratings)} ratings")
                    else:
                        self.logger.info(f"Batch guardado: {len(batch_ratings)} ratings")
                    batch_ratings = []  # Limpar batch
                
                # Delay entre cervejas
                time.sleep(1)
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"❌ Error en {beer['beer_name']}: {e}")
                
                # Detectar erro de conexão do driver
                if "Connection refused" in error_msg or "Failed to establish" in error_msg or "HTTPConnectionPool" in error_msg:
                    self.logger.error("🚨 ERRO CRÍTICO: Driver caiu! Parando execução.")
                    self.logger.error("💡 Solução: Reinicializar o script para reconectar o driver.")
                    # NÃO salvar checkpoint quando há erro de driver
                    return all_ratings
                
                continue
        
        # Mostrar resumen final
        final_total_ratings = self.get_total_ratings_count()
        self.logger.info(f"Extracción finalizada: {len(all_ratings)} ratings nuevos recolectados")
        self.logger.info(f"Total acumulado: {final_total_ratings} ratings de {current_processed_count} cervejas procesadas")
        return all_ratings
    
    def load_beers(self):
        """Cargar cervejas del CSV"""
        beers = []
        with open(self.beers_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                beers.append(row)
        return beers
    
    def collect_beer_ratings(self, beer, max_pages=5):
        """Recolecta ratings de uma cerveja"""
        beer_url = beer['url']
        beer_id = beer['beer_id']
        beer_name = beer['beer_name']
        
        try:
            # Ir a la página de la cerveja
            self.driver.get(beer_url)
            time.sleep(3)
            
            # Hacer clic en "Show More" para cargar más ratings
            click_show_more(self.driver, max_pages, page_type="beer")
            
            # Extrair ratings de la página
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            rating_elements = soup.find_all('div', class_='item')
            
            ratings = []
            for rating_element in rating_elements:
                rating_data = self.extract_rating_data(rating_element, beer_id, beer_name)
                if rating_data:
                    ratings.append(rating_data)
            
            return ratings
            
        except Exception as e:
            self.logger.error(f"Error en ratings de {beer_name}: {e}")
            return []
    
    def extract_rating_data(self, rating_element, beer_id, beer_name):
        """Extrae datos de um checkin/rating baseado na estrutura HTML da imagem"""
        try:
            # User ID e Username (da URL /user/user_id)
            user_id = ''
            username = ''
            user_elem = rating_element.find('a', class_='user')
            if user_elem:
                href = user_elem.get('href', '')
                # Extrair user_id da URL: /user/rogerio_albano_8166
                user_id_match = re.search(r'/user/([^/]+)', href)
                if user_id_match:
                    user_id = user_id_match.group(1)
                
                username = user_elem.get_text(strip=True)
            
            # Rating (solo extrair si existe - algunos checkins no tienen rating)
            rating = ''
            rating_elem = rating_element.find('div', class_='caps')
            if rating_elem:
                rating = rating_elem.get('data-rating', '')
            
            # Si no tiene rating, saltar este checkin
            if not rating:
                return None
            
            # Review text (comentario del checkin)
            review_text = ''
            comment_elem = rating_element.find('div', class_='checkin-comment')
            if comment_elem:
                review_text = comment_elem.get_text(strip=True)
            
            # Serving method (Bottle, Draft, Can, etc.)
            serving_method = ''
            serving_elem = rating_element.find('p', class_='serving')
            if serving_elem:
                # Extrair texto do serving, pode ter uma imagem + texto
                serving_text = serving_elem.get_text(strip=True)
                # Limpiar posibles caracteres extra
                serving_method = re.sub(r'[^\w\s]', '', serving_text).strip()
            
            # Data do checkin 
            date = ''
            date_elem = rating_element.find('a', class_=lambda x: x and 'time' in x)
            if date_elem:
                date = date_elem.get_text(strip=True)
            
            # Fallback: buscar cualquier link con patrón de fecha
            if not date:
                all_links = rating_element.find_all('a')
                for link in all_links:
                    text = link.get_text(strip=True)
                    if re.search(r'\w+,\s+\d{1,2}\s+\w+\s+\d{4}', text):
                        date = text
                        break
            
            # Venue (lugar donde fue hecho el checkin, si disponible)
            venue = ''
            # Procurar por venue no texto do checkin
            text_elem = rating_element.find('p', class_='text')
            if text_elem:
                text_content = text_elem.get_text()
                # Buscar por patrón "at [venue]"
                venue_match = re.search(r'at\s+([^\.]+)', text_content)
                if venue_match:
                    venue = venue_match.group(1).strip()
            
            # Validar datos mínimos (necesita tener user_id, username y rating)
            if not user_id or not username or not rating:
                return None
            
            return {
                'beer_id': beer_id,
                'beer_name': beer_name,
                'user_id': user_id,
                'username': username,
                'rating': rating,
                'review_text': review_text,
                'serving_method': serving_method,
                'venue': venue,
                'date': date
            }
            
        except Exception as e:
            self.logger.error(f"Error extrayendo rating: {e}")
            return None
    
    def save_to_csv(self, ratings):
        """Guarda los ratings en CSV"""
        if not ratings:
            self.logger.warning("No hay datos para guardar")
            return
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'beer_id', 'beer_name', 'user_id', 'username', 'rating', 
                'review_text', 'serving_method', 'venue', 'date'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for rating in ratings:
                writer.writerow(rating)
        
        self.logger.info(f"Datos guardados en {self.output_file}")
    
    def save_batch_to_csv(self, ratings, append=False):
        """Guarda un batch de ratings en CSV (append o overwrite)"""
        if not ratings:
            return
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Determinar si escribir header
        write_header = not append or not os.path.exists(self.output_file)
        
        with open(self.output_file, 'a' if append else 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'beer_id', 'beer_name', 'user_id', 'username', 'rating', 
                'review_text', 'serving_method', 'venue', 'date'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
            
            for rating in ratings:
                writer.writerow(rating)
            
            # Forzar sincronización del archivo
            csvfile.flush()
            os.fsync(csvfile.fileno())
    
    def load_checkpoint(self):
        """Carga el checkpoint para saber desde dónde continuar"""
        if not os.path.exists(self.checkpoint_file):
            return 0
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                return checkpoint_data.get('last_processed_index', 0)
        except Exception as e:
            self.logger.warning(f"Error cargando checkpoint: {e}")
            return 0
    
    def save_checkpoint(self, index):
        """Guarda el checkpoint con el índice actual"""
        try:
            checkpoint_data = {
                'last_processed_index': index,
                'timestamp': time.time(),
                'output_file': self.output_file
            }
            
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Error guardando checkpoint: {e}")
    
    def clear_checkpoint(self):
        """Limpia el checkpoint al finalizar"""
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                self.logger.info("Checkpoint limpiado")
        except Exception as e:
            self.logger.warning(f"Error limpiando checkpoint: {e}")
    
    def get_progress_info(self):
        """Obtiene información del progreso actual"""
        if not os.path.exists(self.checkpoint_file):
            return None
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                return checkpoint_data
        except Exception as e:
            self.logger.warning(f"Error obteniendo info de progreso: {e}")
            return None
    
    def get_already_scraped_beers(self):
        """Obtiene lista de cervejas ya procesadas basado en ratings existentes"""
        already_scraped = set()
        
        if not os.path.exists(self.output_file):
            return already_scraped
        
        try:
            # Leer el archivo de ratings existente
            with open(self.output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    already_scraped.add(row['beer_id'])
            
            # Solo loggear si el número cambió significativamente
            current_count = len(already_scraped)
            if not hasattr(self, '_last_beer_count') or current_count != self._last_beer_count:
                self.logger.info(f"Encontradas {current_count} cervejas ya procesadas")
                self._last_beer_count = current_count
            
            return already_scraped
            
        except Exception as e:
            self.logger.warning(f"Error leyendo cervejas ya procesadas: {e}")
            return already_scraped
    
    def get_detailed_progress_info(self):
        """Obtiene información detallada del progreso incluyendo cervejas ya procesadas"""
        progress_info = self.get_progress_info()
        already_scraped = self.get_already_scraped_beers()
        
        if progress_info:
            progress_info['already_scraped_beers'] = list(already_scraped)
            progress_info['already_scraped_count'] = len(already_scraped)
        
        return progress_info
    
    def get_beer_original_index(self, beer, beers_list):
        """Obtiene el índice original de una cerveja en la lista completa"""
        for idx, b in enumerate(beers_list):
            if b['beer_id'] == beer['beer_id']:
                return idx
        return -1
    
    def get_total_ratings_count(self):
        """Obtiene el número total de ratings en el archivo"""
        if not os.path.exists(self.output_file):
            return 0
        
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                return sum(1 for line in f) - 1  # -1 para excluir el header
        except Exception as e:
            self.logger.warning(f"Error contando ratings totales: {e}")
            return 0
    
    def find_next_beer_to_process(self, popular_beers, already_scraped_beers):
        """Encuentra el próximo índice de cerveja a procesar"""
        for i, beer in enumerate(popular_beers):
            if beer['beer_id'] not in already_scraped_beers:
                return i
        return len(popular_beers)  # Todas ya procesadas
    
    def close(self):
        """Cierra el driver de Selenium"""
        if self.driver:
            self.driver.quit()
            self.logger.info("Driver cerrado")

def main():
    """Función principal"""
    scraper = RatingScraper()
    
    try:
        # Recolectar ratings (teste com 3 cervejas populares)
        ratings = scraper.collect_all_ratings(
            max_beers=3, 
            min_ratings=1000, 
            max_clicks_per_beer=3
        )
        
        # Guardar en CSV
        scraper.save_to_csv(ratings)
        
    finally:
        # Cerrar driver
        scraper.close()

if __name__ == "__main__":
    main()