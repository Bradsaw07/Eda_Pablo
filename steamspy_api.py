"""
Módulo para interactuar con la API de SteamSpy
"""
import requests
import time
import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import config
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class SteamSpyAPI:
    def __init__(self):
        self.base_url = config.API_CONFIG['steamspy']['base_url']
        self.rate_limit = config.API_CONFIG['steamspy']['rate_limit']
        self.last_request = 0
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.steam_charts_url = "https://store.steampowered.com/charts/top-sellers"
        self.api_delay = 1.0  # Delay entre llamadas API (segundos)
        self.cache_dir = Path("data/raw/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _respect_rate_limit(self):
        """Respetar el límite de tasa de la API"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    def get_app_data(self, app_id: int, retries: int = 3) -> Optional[Dict]:
        """Obtener datos de ventas estimadas y estadísticas para un juego"""
        for attempt in range(retries):
            self._respect_rate_limit()
            
            try:
                params = {
                    'request': 'appdetails',
                    'appid': app_id
                }
                response = requests.get(self.base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and data.get('name') != 'Unknown App':
                        try:
                            price = float(data.get('price', 0))
                            initial_price = float(data.get('initialprice', 0))
                            
                            sales_data = {
                                'app_id': app_id,
                                'name': data.get('name', 'Unknown'),
                                'owners': self._parse_owners(data.get('owners', '0')),
                                'average_playtime': int(data.get('average_forever', 0)),
                                'median_playtime': int(data.get('median_forever', 0)),
                                'price': price / 100 if price else 0,
                                'initial_price': initial_price / 100 if initial_price else 0,
                                'discount': float(data.get('discount', 0)),
                                'ccu': int(data.get('ccu', 0)),
                                'peak_ccu': int(data.get('peak_ccu', 0)),
                                'score_rank': str(data.get('score_rank', '')),
                                'positive': int(data.get('positive', 0)),
                                'negative': int(data.get('negative', 0)),
                                'timestamp': datetime.now()
                            }
                            
                            sales_data['estimated_revenue'] = sales_data['owners'] * sales_data['price']
                            logger.info(f"Datos obtenidos para {sales_data['name']}")
                            return sales_data
                        
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error procesando datos para app {app_id}: {e}")
                            if attempt < retries - 1:
                                continue
                
                if attempt < retries - 1:
                    logger.warning(f"Reintento {attempt + 1} para app {app_id}")
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error obteniendo datos de SteamSpy para app {app_id}: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                    
        return None

    def _parse_owners(self, owners_str: str) -> int:
        """Convierte el rango de propietarios a un número estimado"""
        try:
            if not owners_str:
                return 0
            if '..' in owners_str:
                low, high = map(lambda x: int(x.strip().replace(',', '')), owners_str.split('..'))
                return (low + high) // 2
            return int(owners_str.replace(',', ''))
        except (ValueError, TypeError):
            logger.warning(f"No se pudo parsear el número de propietarios: {owners_str}")
            return 0

    def _load_existing_games(self) -> Dict[int, Dict]:
        """Carga los juegos existentes de los archivos CSV"""
        existing_games = {}
        try:
            for file in self.data_dir.glob("top_games_*.csv"):
                df = pd.read_csv(file)
                for _, row in df.iterrows():
                    app_id = int(row['app_id'])
                    if app_id not in existing_games or pd.isna(existing_games[app_id].get('owners', 0)):
                        existing_games[app_id] = row.to_dict()
            logger.info(f"Juegos existentes cargados: {len(existing_games)}")
            return existing_games
        except Exception as e:
            logger.error(f"Error cargando juegos existentes: {e}")
            return {}

    def get_top_games(self, limit: int = 100) -> List[Dict]:
        """Obtener los juegos más populares según SteamSpy"""
        self._respect_rate_limit()
        
        try:
            # Cargar juegos existentes
            existing_games = self._load_existing_games()
            logger.info(f"Juegos existentes encontrados: {len(existing_games)}")
            
            # Obtener nuevos datos
            endpoints = [
                {'request': 'top100in2weeks'},
                {'request': 'top100forever'},
                {'request': 'top100owned'}
            ]
            
            new_games = []
            for params in endpoints:
                response = requests.get(self.base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    for app_id, game_data in data.items():
                        app_id = int(app_id)
                        try:
                            price = float(game_data.get('price', 0))
                            game_info = {
                                'app_id': app_id,
                                'name': game_data.get('name', 'Unknown'),
                                'owners': self._parse_owners(game_data.get('owners', '0')),
                                'players_2weeks': int(game_data.get('players_2weeks', 0)),
                                'peak_ccu': int(game_data.get('peak_ccu', 0)),
                                'price': price / 100 if price else 0,
                                'timestamp': datetime.now()
                            }
                            
                            # Actualizar o agregar nuevo juego
                            if app_id in existing_games:
                                # Actualizar solo campos vacíos o nulos
                                for key, value in game_info.items():
                                    if key not in existing_games[app_id] or pd.isna(existing_games[app_id][key]):
                                        existing_games[app_id][key] = value
                                logger.info(f"Actualizado juego existente: {game_info['name']}")
                            else:
                                existing_games[app_id] = game_info
                                logger.info(f"Agregado nuevo juego: {game_info['name']}")
                                
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error procesando juego {app_id}: {e}")
                            continue
                
                self._respect_rate_limit()
            
            # Convertir diccionario a lista y ordenar
            all_games = list(existing_games.values())
            sorted_games = sorted(all_games, key=lambda x: x.get('owners', 0), reverse=True)
            
            # Tomar los top N juegos
            final_games = sorted_games[:limit]
            
            if final_games:
                # Guardar solo si hay nuevos datos
                self._save_to_csv(final_games, 'top_games')
                logger.info(f"Total de juegos guardados: {len(final_games)}")
            
            return final_games
            
        except Exception as e:
            logger.error(f"Error obteniendo top games de SteamSpy: {str(e)}")
            return []

    def _save_to_csv(self, data: List[Dict], prefix: str):
        """Guarda los datos en un archivo CSV"""
        if not data:
            return
            
        df = pd.DataFrame(data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"{prefix}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Datos guardados en {filename}")

    def get_additional_games(self, min_games: int = 200) -> List[Dict]:
        """
        Obtiene juegos adicionales usando diferentes criterios para enriquecer la base de datos
        """
        logger.info("Buscando juegos adicionales para enriquecer la base de datos...")
        
        # Cargar juegos existentes
        existing_games = self._load_existing_games()
        existing_ids = set(existing_games.keys())
        logger.info(f"Juegos existentes: {len(existing_ids)}")
        
        # Diferentes endpoints para obtener juegos variados
        discovery_endpoints = [
            {'request': 'all', 'page': 0},  # Todos los juegos (paginado)
            {'request': 'genre', 'genre': 'RPG'},
            {'request': 'genre', 'genre': 'Action'},
            {'request': 'genre', 'genre': 'Strategy'},
            {'request': 'genre', 'genre': 'Indie'},
            {'request': 'tag', 'tag': 'Multiplayer'},
            {'request': 'tag', 'tag': 'Singleplayer'},
            {'request': 'top100forever'},
            {'request': 'hot100'}
        ]
        
        new_games = {}
        total_attempts = 0
        max_attempts = 50  # Límite de intentos para evitar bucles infinitos
        
        while len(new_games) < min_games and total_attempts < max_attempts:
            for endpoint in discovery_endpoints:
                self._respect_rate_limit()
                
                try:
                    # Añadir página aleatoria para el endpoint 'all'
                    if endpoint['request'] == 'all':
                        endpoint['page'] = total_attempts
                    
                    logger.info(f"Consultando endpoint: {endpoint}")
                    response = requests.get(self.base_url, params=endpoint)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for app_id, game_data in data.items():
                            try:
                                app_id = int(app_id)
                                
                                # Saltar juegos que ya tenemos
                                if app_id in existing_ids or app_id in new_games:
                                    continue
                                
                                # Verificar que sea un juego válido
                                if game_data.get('name') == 'Unknown App':
                                    continue
                                    
                                # Obtener datos detallados del juego
                                detailed_data = self.get_app_data(app_id)
                                if detailed_data:
                                    new_games[app_id] = detailed_data
                                    logger.info(f"Nuevo juego encontrado: {detailed_data['name']}")
                                    
                                    # Guardar progreso cada 10 juegos nuevos
                                    if len(new_games) % 10 == 0:
                                        self._save_progress(list(new_games.values()))
                                        
                                    if len(new_games) >= min_games:
                                        break
                            
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Error procesando juego {app_id}: {e}")
                                continue
                            
                except Exception as e:
                    logger.error(f"Error con endpoint {endpoint}: {e}")
                    continue
                
                if len(new_games) >= min_games:
                    break
                    
            total_attempts += 1
            logger.info(f"Intento {total_attempts}: {len(new_games)} nuevos juegos encontrados")
        
        # Guardar todos los juegos nuevos
        new_games_list = list(new_games.values())
        if new_games_list:
            self._save_to_csv(new_games_list, 'additional_games')
            logger.info(f"Total de nuevos juegos encontrados: {len(new_games_list)}")
        else:
            logger.warning("No se encontraron nuevos juegos")
        
        return new_games_list

    def _save_progress(self, games: List[Dict], prefix: str = 'progress'):
        """Guarda el progreso de la recolección de datos"""
        if not games:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"{prefix}_{timestamp}.csv"
        
        try:
            df = pd.DataFrame(games)
            df.to_csv(filename, index=False)
            logger.info(f"Progreso guardado en: {filename}")
        except Exception as e:
            logger.error(f"Error guardando progreso: {e}")

    def _save_and_update_games(self, new_games: List[Dict], filename: str = "steam_games_database.csv"):
        """Guarda y actualiza la base de datos de juegos"""
        try:
            # Cargar datos existentes si el archivo existe
            if file_path.exists():
                existing_df = pd.read_csv(file_path)
                logger.info(f"Base de datos existente cargada: {len(existing_df)} juegos")
            else:
                existing_df = pd.DataFrame()
                logger.info("Creando nueva base de datos")
            
            # Convertir nuevos juegos a DataFrame
            new_df = pd.DataFrame(new_games)
            
            if not existing_df.empty:
                # Actualizar registros existentes y añadir nuevos
                combined_df = pd.concat([existing_df, new_df])
                # Eliminar duplicados manteniendo la información más reciente
                combined_df = combined_df.sort_values('timestamp').drop_duplicates(
                    subset=['app_id'], keep='last'
                )
            else:
                combined_df = new_df
            
            # Calcular positive_ratio
            combined_df = self.create_positive_ratio(combined_df)
            
            # Guardar DataFrame actualizado
            combined_df.to_csv(file_path, index=False)
            logger.info(f"Base de datos actualizada guardada: {len(combined_df)} juegos totales")
            
            # Mostrar estadísticas de completitud
            self._show_data_completeness(combined_df)
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error guardando/actualizando base de datos: {e}")
            raise

    def _show_data_completeness(self, df: pd.DataFrame):
        """Muestra estadísticas sobre la completitud de los datos"""
        logger.info("\nEstadísticas de completitud de datos:")
        total_games = len(df)
        
        completeness = {
            'Metacritic Score': len(df[df['metacritic_score'] > 0]),
            'User Score': len(df[df['user_score'] > 0]),
            'Precio': len(df[df['price'].notna()]),
            'Propietarios': len(df[df['owners'] > 0]),
            'Reseñas': len(df[df['total_reviews'] > 0]),
            'Géneros': len(df[df['genres'].notna() & (df['genres'] != '')]),
            'Fecha de lanzamiento': len(df[df['release_date'].notna()])
        }
        
        for field, count in completeness.items():
            percentage = (count / total_games) * 100
            logger.info(f"{field}: {count}/{total_games} ({percentage:.1f}%)")

    def get_metacritic_score(self, app_id: int) -> tuple:
        """Obtiene puntuación de Metacritic y usuarios de múltiples fuentes"""
        try:
            # Intentar obtener de Steam API
            steam_url = f"https://store.steampowered.com/api/appdetails?appid={app_id}&cc=us&l=en"
            response = requests.get(steam_url)
            if response.status_code == 200:
                data = response.json()
                if data and str(app_id) in data and data[str(app_id)].get('success'):
                    metacritic = data[str(app_id)]['data'].get('metacritic', {})
                    if metacritic:
                        return metacritic.get('score', 0), metacritic.get('url', '')
            
            # Si no hay datos de Steam, intentar web scraping de Metacritic
            # Nota: Implementar con beautifulsoup4 si es necesario
            
            return 0, ''
            
        except Exception as e:
            logger.error(f"Error obteniendo puntuación Metacritic para {app_id}: {e}")
            return 0, ''

    def collect_comprehensive_data(self, target_total: int = 2000):
        """Recolecta datos comprensivos de juegos incluyendo géneros"""
        start_time = time.time()
        games_data = []
        
        try:
            page = 0
            errors = 0
            successful_requests = 0
            retries = 0
            
            while len(games_data) < target_total and page < 100:
                logger.info(f"\nProcesando página {page + 1}...")
                
                # Obtener datos de la API
                params = {
                    'request': 'all',
                    'page': page
                }
                
                response = requests.get(self.base_url, params=params)
                if response.status_code == 200:
                    games_batch = response.json()
                    logger.info(f"Encontrados {len(games_batch)} juegos en la página {page + 1}")
                    
                    for app_id, basic_data in games_batch.items():
                        if len(games_data) >= target_total:
                            break
                        
                        try:
                            # Obtener datos detallados
                            detailed_data = self._get_detailed_game_data(app_id)
                            if detailed_data:
                                # Obtener datos adicionales
                                review_data = self.get_review_data(app_id)
                                ea_data = self.get_early_access_data(app_id)
                                metacritic_data = self.get_metacritic_data(app_id)
                                dlc_data = self.get_dlc_data(app_id)
                                microtrans_data = self.get_microtransactions_data(app_id)
                                
                                # Procesar precio de manera segura
                                try:
                                    raw_price = detailed_data.get('price')
                                    if raw_price is not None:
                                        price = float(raw_price) / 100
                                    else:
                                        price = 0.0
                                except (ValueError, TypeError):
                                    price = 0.0
                                    logger.warning(f"Error procesando precio para juego {app_id}")
                                
                                # Determinar si es gratuito
                                is_free = price == 0
                                
                                # Crear game_info con todos los datos
                                game_info = {
                                    'app_id': app_id,
                                    'name': detailed_data.get('name', 'Unknown'),
                                    'price': price,
                                    'is_free': is_free,
                                    'owners': self._parse_owners(detailed_data.get('owners', '0')),
                                    'positive': int(detailed_data.get('positive', 0)),
                                    'negative': int(detailed_data.get('negative', 0)),
                                    'average_playtime': int(detailed_data.get('average_forever', 0)),
                                    'median_playtime': int(detailed_data.get('median_forever', 0)),
                                    'genres': self._classify_game_genre(detailed_data),
                                    'initial_price': float(detailed_data.get('initial_price', 0)) / 100,
                                    'discount': float(detailed_data.get('discount', 0)),
                                    'ccu': int(detailed_data.get('ccu', 0)),
                                    'peak_ccu': int(detailed_data.get('peak_ccu', 0)),
                                    'score_rank': str(detailed_data.get('score_rank', '')),
                                    'tags': detailed_data.get('tags', []),
                                    'release_date': detailed_data.get('release_date', ''),
                                    'is_indie': self._classify_as_indie(
                                        price=price,
                                        tags=detailed_data.get('tags', []),
                                        name=detailed_data.get('name', ''),
                                        owners=self._parse_owners(detailed_data.get('owners', '0')),
                                        developer=detailed_data.get('developer', ''),
                                        publisher=detailed_data.get('publisher', '')
                                    ),
                                    'current_players': self.get_current_players(app_id),
                                    'review_count': review_data['review_count'],
                                    'review_score': review_data['review_score'],
                                    'metacritic_score': metacritic_data['metacritic_score'],
                                    'user_score': metacritic_data['user_score'],
                                    'critic_reviews_count': metacritic_data['critic_reviews_count'],
                                    'dlc_count': dlc_data['dlc_count'],
                                    'has_dlc': dlc_data['has_dlc'],
                                    'has_microtransactions': microtrans_data['has_microtransactions'],
                                    'microtransaction_type': microtrans_data['microtransaction_type'],
                                    'early_access_date': ea_data['early_access_date'],
                                    'full_release_date': ea_data['full_release_date']
                                }
                                
                                games_data.append(game_info)
                                successful_requests += 1
                                
                                # Mostrar progreso
                                if len(games_data) % 10 == 0:
                                    self._show_progress(games_data, start_time, target_total, successful_requests, errors, retries)
                    
                        except Exception as e:
                            errors += 1
                            logger.warning(f"Error procesando juego {app_id}: {str(e)}")
                            continue
                
                page += 1
                time.sleep(self.api_delay)
            
            # Convertir a DataFrame y mostrar resumen final
            df = pd.DataFrame(games_data)
            
            # Calcular positive_ratio
            df = self.create_positive_ratio(df)
            
            # Guardar datos
            output_file = self.data_dir / "steam_games_database.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"\nDatos guardados en: {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error en recolección de datos: {e}")
            raise

    def _classify_game_genre(self, game_data: Dict) -> str:
        """Clasifica los géneros del juego basado en múltiples factores"""
        try:
            name = str(game_data.get('name', '')).lower()
            tags = [tag.lower() for tag in game_data.get('tags', [])]
            price = float(game_data.get('price', 0)) / 100
            playtime = int(game_data.get('average_forever', 0))
            
            # Diccionario de palabras clave por género
            genre_keywords = {
                'Action': ['action', 'shooter', 'fps', 'fighting', 'battle', 'combat', 'warfare', 'gun'],
                'Adventure': ['adventure', 'exploration', 'story-rich', 'narrative', 'quest', 'journey'],
                'RPG': ['rpg', 'role-playing', 'jrpg', 'wrpg', 'fantasy', 'dragon', 'souls', 'dungeon'],
                'Strategy': ['strategy', 'rts', 'turn-based', 'grand strategy', 'tactics', 'tower defense'],
                'Simulation': ['simulation', 'simulator', 'tycoon', 'management', 'farming', 'life sim'],
                'Sports': ['sports', 'racing', 'football', 'basketball', 'soccer', 'baseball', 'nba', 'fifa'],
                'Casual': ['casual', 'puzzle', 'match-3', 'family-friendly', 'educational', 'point-and-click'],
                'MMO': ['mmo', 'mmorpg', 'multiplayer online', 'online rpg', 'massive multiplayer'],
                'Horror': ['horror', 'survival horror', 'psychological horror', 'zombie', 'dark', 'gore'],
                'Indie': ['indie', 'experimental', 'artistic', 'minimalist', 'retro', 'pixel']
            }
            
            assigned_genres = []
            
            # 1. Primero verificar tags oficiales de Steam
            steam_genres = game_data.get('genres', [])
            if steam_genres and isinstance(steam_genres, list):
                for genre in steam_genres:
                    if isinstance(genre, dict) and 'description' in genre:
                        assigned_genres.append(genre['description'])
            
            # 2. Si no hay géneros de Steam, buscar en tags y nombre
            if not assigned_genres:
                for genre, keywords in genre_keywords.items():
                    if any(keyword in tags for keyword in keywords) or any(keyword in name for keyword in keywords):
                        if genre not in assigned_genres:
                            assigned_genres.append(genre)
            
            # 3. Clasificación basada en características
            if not assigned_genres:
                if playtime > 1000:
                    if price > 30:
                        assigned_genres.append('RPG')
                    else:
                        assigned_genres.append('MMO')
                
                if price == 0:
                    if playtime > 500:
                        assigned_genres.append('MMO')
                    else:
                        assigned_genres.append('Casual')
                
                if price > 40:
                    assigned_genres.append('RPG')
            
            # 4. Asegurar al menos un género
            if not assigned_genres:
                if price > 30:
                    assigned_genres.append('Action')
                elif price > 15:
                    assigned_genres.append('Indie')
                else:
                    assigned_genres.append('Casual')
            
            # 5. Reglas de combinación especiales
            if 'Action' in assigned_genres and 'RPG' in assigned_genres:
                assigned_genres.append('Action RPG')
            
            if 'Horror' in assigned_genres and 'Survival' in assigned_genres:
                assigned_genres.append('Survival Horror')
            
            # Limitar a 3 géneros principales y formatear
            return ', '.join(assigned_genres[:3])
        
        except Exception as e:
            logger.error(f"Error clasificando género: {e}")
            return 'Unclassified'  # Valor por defecto en caso de error

    def _get_detailed_game_data(self, app_id: str) -> Optional[Dict]:
        """Obtiene datos detallados de un juego incluyendo tags y géneros"""
        max_retries = 3
        base_delay = 2  # 2 segundos de espera base
        
        for attempt in range(max_retries):
            try:
                # Obtener datos básicos de SteamSpy
                params = {
                    'request': 'appdetails',
                    'appid': app_id
                }
                response = requests.get(self.base_url, params=params)
                if response.status_code != 200:
                    logger.warning(f"Intento {attempt + 1}: Error en SteamSpy API para {app_id}")
                    time.sleep(base_delay * (attempt + 1))  # Delay exponencial
                    continue
                
                game_data = response.json()
                
                # Esperar antes de hacer la segunda petición
                time.sleep(base_delay)
                
                # Obtener datos adicionales de Steam Store con manejo de rate limit
                store_data = self._get_steam_store_data_with_retry(app_id, max_retries, base_delay)
                if store_data:
                    game_data.update(store_data)
                
                return game_data
                
            except Exception as e:
                logger.error(f"Error en intento {attempt + 1} para {app_id}: {e}")
                time.sleep(base_delay * (attempt + 1))
                
        return None

    def _get_steam_store_data_with_retry(self, app_id: str, max_retries: int = 3, base_delay: int = 2) -> Optional[Dict]:
        """Obtiene datos de Steam Store con sistema de reintentos y delays"""
        for attempt in range(max_retries):
            try:
                # Añadir delay exponencial entre intentos
                if attempt > 0:
                    delay = base_delay * (2 ** attempt)  # Delay exponencial: 2s, 4s, 8s...
                    logger.info(f"Esperando {delay} segundos antes del reintento {attempt + 1}")
                    time.sleep(delay)
                
                steam_url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
                response = requests.get(steam_url)
                
                if response.status_code == 429:  # Too Many Requests
                    logger.warning(f"Rate limit alcanzado en intento {attempt + 1}. Esperando...")
                    continue
                
                if response.status_code == 200:
                    data = response.json()
                    if data and str(app_id) in data and data[str(app_id)].get('success'):
                        return data[str(app_id)]['data']
                    
            except Exception as e:
                logger.error(f"Error obteniendo datos de Steam Store para {app_id}: {e}")
                
        return None

    def cleanup_temp_files(self):
        """Limpia archivos temporales manteniendo solo steam_games_database.csv"""
        logger.info("Limpiando archivos temporales...")
        
        try:
            main_db = self.data_dir / "steam_games_database.csv"
            if not main_db.exists():
                logger.error("No se encontró la base de datos principal")
                return
                
            # Listar todos los archivos CSV excepto steam_games_database.csv
            temp_files = [f for f in self.data_dir.glob("*.csv") 
                         if f.name != "steam_games_database.csv"]
            
            # Eliminar archivos temporales
            for file in temp_files:
                try:
                    file.unlink()
                    logger.info(f"Archivo temporal eliminado: {file.name}")
                except Exception as e:
                    logger.error(f"Error eliminando {file.name}: {e}")
                    
            logger.info("Limpieza de archivos temporales completada")
            
        except Exception as e:
            logger.error(f"Error durante la limpieza: {e}")

    def get_game_details(self, appid: str) -> Dict:
        """Obtiene detalles de un juego específico"""
        try:
            url = f"{self.base_url}/api.php"
            params = {
                'request': 'appdetails',
                'appid': str(appid)  # Aseguramos que appid sea string
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convertir owners string a rango numérico
            owners_str = data.get('owners', '0 - 0')
            owners_range = [int(x.strip()) for x in owners_str.split('-')]
            owners_avg = sum(owners_range) // 2
            
            # Extraer y limpiar géneros
            genres = data.get('genre', '').split(',')
            genres = [genre.strip() for genre in genres if genre.strip()]
            
            return {
                'appid': int(appid),
                'name': data.get('name', ''),
                'owners': owners_avg,
                'genres': ','.join(genres),  # Guardamos géneros como string separado por comas
                'price': float(data.get('price', 0)),
                'positive_ratio': float(data.get('positive_ratio', 0)),
                'negative_ratio': float(data.get('negative_ratio', 0)),
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo detalles para appid {appid}: {e}")
            return None

    def get_game_details(self, appid: int) -> Optional[Dict]:
        """Obtiene detalles detallados de un juego desde Steam API"""
        try:
            params = {
                'appids': appid,
                'cc': 'us',
                'l': 'english'
            }
            response = requests.get(self.steam_api_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if str(appid) in data and data[str(appid)]['success']:
                    return data[str(appid)]['data']
            return None
        except Exception as e:
            logger.error(f"Error obteniendo detalles de Steam para appid {appid}: {e}")
            return None

    def update_game_data(self, csv_path: str):
        """Actualiza los datos del CSV con información de Steam"""
        try:
            # Cargar el CSV existente
            df = pd.read_csv(csv_path)
            
            # Crear copias de respaldo
            backup_path = Path(csv_path).parent / f"backup_{Path(csv_path).name}"
            df.to_csv(backup_path, index=False)
            
            # Actualizar registros con campos faltantes o cero
            for index, row in df.iterrows():
                if (pd.isna(row['genres']) or 
                    pd.isna(row['metacritic_score']) or 
                    row['price'] == 0 or 
                    pd.isna(row['price'])):
                    
                    # Obtener datos actualizados
                    steam_data = self.get_game_details(row['appid'])
                    if steam_data:
                        # Actualizar géneros
                        if 'genres' in steam_data:
                            genres = ','.join([g['description'] for g in steam_data['genres']])
                            df.at[index, 'genres'] = genres
                        
                        # Actualizar metacritic
                        if 'metacritic' in steam_data:
                            df.at[index, 'metacritic_score'] = steam_data['metacritic']['score']
                        
                        # Actualizar precio
                        if 'price_overview' in steam_data:
                            df.at[index, 'price'] = steam_data['price_overview']['final'] / 100
                    
                    time.sleep(self.delay)  # Respetar límites de rate
            
            # Guardar datos actualizados
            output_path = Path(csv_path).parent / f"updated_{Path(csv_path).name}"
            df.to_csv(output_path, index=False)
            logger.info(f"Datos actualizados guardados en: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error actualizando datos: {e}")
            raise

    def get_genre_data(self) -> Dict[str, list]:
        """Obtiene datos organizados por género de Steam Charts"""
        genres = {
            'Action': ['action', 'arcade', 'fighting', 'fps', 'hack_and_slash'],
            'Adventure': ['adventure', 'metroidvania', 'visual_novel'],
            'RPG': ['rpg', 'jrpg', 'action_rpg', 'strategy_rpg'],
            'Simulation': ['simulation', 'building', 'farming', 'life_sim'],
            'Strategy': ['strategy', 'rts', 'turn_based', 'grand_strategy'],
            'Sports': ['sports', 'racing', 'sports_sim'],
            'Indie': ['indie'],
            'MMO': ['mmo', 'mmorpg'],
            'Horror': ['horror', 'survival_horror'],
            'Puzzle': ['puzzle', 'logic']
        }
        
        genre_data = {genre: [] for genre in genres}
        
        try:
            for genre, subgenres in genres.items():
                for subgenre in subgenres:
                    params = {
                        'request': 'genre',
                        'genre': subgenre
                    }
                    response = requests.get(self.base_url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        genre_data[genre].extend(data.values())
                    time.sleep(self.api_delay)
            
            return genre_data
        except Exception as e:
            logger.error(f"Error obteniendo datos por género: {e}")
            return {}

    def get_top_games(self, category: str = 'top-sellers') -> list:
        """Obtiene los juegos más populares de Steam Charts"""
        valid_categories = ['top-sellers', 'most-played', 'trending']
        if category not in valid_categories:
            raise ValueError(f"Categoría inválida. Debe ser una de: {valid_categories}")
        
        url = f"https://store.steampowered.com/charts/{category}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            # Aquí se implementaría el parsing real de la página
            # Este es un ejemplo simplificado
            return []
        except Exception as e:
            logger.error(f"Error obteniendo top games: {e}")
            return []

    def analyze_early_access_impact(self) -> pd.DataFrame:
        """Analiza el impacto del acceso anticipado"""
        try:
            early_access_games = []
            regular_games = []
            
            # Obtener muestra de juegos
            params = {'request': 'all'}
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                games = response.json()
                
                for app_id, game_data in games.items():
                    detailed_data = self.get_app_data(int(app_id))
                    if detailed_data:
                        game_info = {
                            'name': game_data.get('name'),
                            'owners': game_data.get('owners'),
                            'positive_ratio': game_data.get('positive_ratio'),
                            'price': game_data.get('price'),
                            'is_early_access': detailed_data.get('is_early_access', False)
                        }
                        
                        if game_info['is_early_access']:
                            early_access_games.append(game_info)
                        else:
                            regular_games.append(game_info)
                
                # Crear DataFrame para análisis
                df_early = pd.DataFrame(early_access_games)
                df_regular = pd.DataFrame(regular_games)
                
                return pd.concat([df_early, df_regular])
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error analizando impacto de acceso anticipado: {e}")
            return pd.DataFrame()

    def get_seasonal_data(self) -> pd.DataFrame:
        """Obtiene datos de lanzamientos por temporada"""
        try:
            games_data = []
            current_year = datetime.now().year
            
            # Obtener datos de los últimos 5 años
            for year in range(current_year - 5, current_year + 1):
                params = {
                    'request': 'all',
                    'year': year
                }
                response = requests.get(self.base_url, params=params)
                if response.status_code == 200:
                    games = response.json()
                    for game in games.values():
                        if 'release_date' in game:
                            games_data.append({
                                'name': game.get('name'),
                                'release_date': game.get('release_date'),
                                'owners': game.get('owners'),
                                'positive_ratio': game.get('positive_ratio'),
                                'price': game.get('price')
                            })
            
            return pd.DataFrame(games_data)
        except Exception as e:
            logger.error(f"Error obteniendo datos estacionales: {e}")
            return pd.DataFrame()

    def _get_steam_store_data(self, app_id: int) -> Optional[Dict[str, Any]]:
        """Obtiene datos adicionales de la Steam Store"""
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if str(app_id) in data and data[str(app_id)].get('success'):
                store_data = data[str(app_id)]['data']
                
                # Manejar géneros de forma segura
                genres = []
                if 'genres' in store_data and store_data['genres']:
                    try:
                        genres = [genre.get('description', '') for genre in store_data['genres']]
                    except (AttributeError, TypeError):
                        genres = ['Unclassified']
                
                # Asegurar que siempre haya un género
                if not genres:
                    genres = ['Unclassified']
                
                return {
                    'genres': ','.join(genres),
                    'is_early_access': store_data.get('early_access', False),
                    'update_history': self._get_update_history(app_id),
                    'metacritic_score': store_data.get('metacritic', {}).get('score', 0),
                    'categories': [cat.get('description', '') for cat in store_data.get('categories', [])],
                    'features': [feat.get('description', '') for feat in store_data.get('features', [])],
                    'dlc_count': len(store_data.get('dlc', [])),
                    'achievements': store_data.get('achievements', {}).get('total', 0),
                    'release_date': store_data.get('release_date', {}).get('date', ''),
                    'supported_languages': store_data.get('supported_languages', ''),
                    'is_free': store_data.get('is_free', False),
                    'price': store_data.get('price_overview', {}).get('final', 0) / 100 if not store_data.get('is_free', False) else 0.0
                }
            return None
        except Exception as e:
            logger.error(f"Error obteniendo datos de Steam Store para {app_id}: {e}")
            return None

    def _get_update_history(self, app_id: int) -> Dict[str, Any]:
        """Obtiene historial de actualizaciones de un juego"""
        try:
            url = f"https://store.steampowered.com/api/appupdates/{app_id}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Contar actualizaciones en el último año
                current_time = datetime.now()
                one_year_ago = current_time - timedelta(days=365)
                
                updates = 0
                last_update = None
                
                if 'updates' in data:
                    for update in data['updates']:
                        update_date = datetime.fromtimestamp(update.get('timestamp', 0))
                        if update_date > one_year_ago:
                            updates += 1
                        if not last_update or update_date > last_update:
                            last_update = update_date
                
                return {
                    'update_count': updates,
                    'last_update': last_update.strftime("%Y-%m-%d") if last_update else None,
                    'updates_per_month': round(updates / 12, 2) if updates > 0 else 0
                }
            
            # Si no podemos obtener datos de actualizaciones, devolver valores por defecto
            return {
                'update_count': 0,
                'last_update': None,
                'updates_per_month': 0
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo historial de actualizaciones para {app_id}: {e}")
            return {
                'update_count': 0,
                'last_update': None,
                'updates_per_month': 0
            }

    def _classify_as_indie(self, price: float, tags: List[str], name: str, 
                          owners: int, developer: str, publisher: str) -> bool:
        """
        Clasifica un juego como indie basado en múltiples criterios
        """
        # Convertir todo a minúsculas para comparaciones
        tags = [tag.lower() for tag in tags]
        name = name.lower()
        developer = developer.lower()
        publisher = publisher.lower()
        
        # Criterios para clasificar como indie
        indie_indicators = {
            'tags': ['indie', 'casual', 'pixel graphics', 'retro', '2d platformer'],
            'price_threshold': 20.0,  # Juegos por debajo de $20
            'owners_threshold': 100000,  # Menos de 100k propietarios
            'major_publishers': ['electronic arts', 'ubisoft', 'activision', 'bethesda', 
                               '2k games', 'square enix', 'bandai namco', 'capcom']
        }
        
        # Sistema de puntos para clasificación
        indie_score = 0
        
        # 1. Verificar tags (3 puntos)
        if any(tag in tags for tag in indie_indicators['tags']):
            indie_score += 3
        
        # 2. Verificar precio (2 puntos)
        if price <= indie_indicators['price_threshold']:
            indie_score += 2
        
        # 3. Verificar número de propietarios (1 punto)
        if owners <= indie_indicators['owners_threshold']:
            indie_score += 1
        
        # 4. Verificar publisher (-3 puntos si es AAA)
        if any(pub in publisher for pub in indie_indicators['major_publishers']):
            indie_score -= 3
        
        # 5. Verificar si desarrollador y publisher son el mismo (2 puntos)
        if developer and developer == publisher:
            indie_score += 2
        
        # Clasificar como indie si score >= 3
        return indie_score >= 3

    def get_metacritic_data(self, app_id: str) -> Dict:
        """Obtiene datos de Metacritic del curator"""
        try:
            url = f"https://store.steampowered.com/curator/35387214-Metacritic./app/{app_id}"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            
            if response and response.status_code == 200 and response.text:  # Verificar que tenemos respuesta válida
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Buscar elementos con manejo de errores
                user_score_elem = soup.find('span', {'class': 'user-score'})
                critic_count_elem = soup.find('span', {'class': 'critic-count'})
                score_elem = soup.find('span', {'class': 'score'})
                
                return {
                    'user_score': float(user_score_elem.text) if user_score_elem else 0.0,
                    'critic_reviews_count': int(critic_count_elem.text) if critic_count_elem else 0,
                    'metacritic_score': int(score_elem.text) if score_elem else 0
                }
            
            # Si no hay respuesta válida, devolver valores por defecto
            logger.warning(f"No se encontraron datos de Metacritic para {app_id}")
            return {
                'user_score': 0.0,
                'critic_reviews_count': 0,
                'metacritic_score': 0
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de Metacritic para {app_id}: {e}")
            return {
                'user_score': 0.0,
                'critic_reviews_count': 0,
                'metacritic_score': 0
            }

    def get_steam_reviews_data(self, app_id: str) -> Dict:
        """Obtiene datos de reseñas y dificultad de Steam"""
        try:
            url = f"https://store.steampowered.com/reviews/?l=spanish&appid={app_id}"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Analizar reseñas para determinar dificultad
                difficulty_keywords = {
                    'Easy': ['fácil', 'sencillo', 'casual', 'simple'],
                    'Medium': ['normal', 'moderado', 'balanced'],
                    'Hard': ['difícil', 'complejo', 'challenging', 'hardcore']
                }
                
                reviews = soup.find_all('div', {'class': 'review_content'})
                difficulty_scores = []
                
                for review in reviews:
                    text = review.get_text().lower()
                    for difficulty, keywords in difficulty_keywords.items():
                        if any(keyword in text for keyword in keywords):
                            difficulty_scores.append(difficulty)
                
                # Determinar dificultad basada en la moda
                difficulty = 'Medium'  # valor por defecto
                if difficulty_scores:
                    from statistics import mode
                    difficulty = mode(difficulty_scores)
                
                # Extraer conteos de reseñas positivas y negativas
                positive = soup.find('div', {'class': 'positive'})
                negative = soup.find('div', {'class': 'negative'})
                
                return {
                    'difficulty': difficulty,
                    'positive_reviews': int(positive.text) if positive else 0,
                    'negative_reviews': int(negative.text) if negative else 0
                }
                
            return {'difficulty': 'Medium', 'positive_reviews': 0, 'negative_reviews': 0}
            
        except Exception as e:
            logger.error(f"Error obteniendo reseñas para {app_id}: {e}")
            return {'difficulty': 'Medium', 'positive_reviews': 0, 'negative_reviews': 0}

    def get_sales_data(self, app_id: str) -> Dict:
        """Obtiene datos de ventas de Steam Charts"""
        try:
            url = "https://store.steampowered.com/charts/mostplayed"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Buscar el juego en la lista de más jugados
                game_row = soup.find('a', {'href': f'/app/{app_id}'})
                if game_row:
                    parent = game_row.parent
                    players = parent.find('span', {'class': 'weeklystats'})
                    
                    return {
                        'current_players': int(players.text.replace(',', '')) if players else 0,
                        'is_top_seller': True
                    }
                
            return {'current_players': 0, 'is_top_seller': False}
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de ventas para {app_id}: {e}")
            return {'current_players': 0, 'is_top_seller': False}

    def get_current_players(self, app_id: str) -> int:
        """Obtiene el número actual de jugadores de Steam Charts"""
        try:
            url = "https://store.steampowered.com/charts/mostplayed"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                game_row = soup.find('a', {'href': f'/app/{app_id}'})
                
                if game_row:
                    players_span = game_row.find_next('span', {'class': 'currentPlayers'})
                    if players_span:
                        return int(players_span.text.replace(',', ''))
            return 0
        except Exception as e:
            logger.error(f"Error obteniendo jugadores actuales para {app_id}: {e}")
            return 0

    def get_review_data(self, app_id: str) -> Dict:
        """Obtiene datos de reseñas de SteamDB"""
        try:
            url = f"https://steamdb.info/app/{app_id}"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                reviews_div = soup.find('div', {'class': 'user-reviews'})
                
                if reviews_div:
                    return {
                        'review_count': int(reviews_div.find('span', {'class': 'total-reviews'}).text),
                        'review_score': float(reviews_div.find('span', {'class': 'score'}).text)
                    }
            return {'review_count': 0, 'review_score': 0}
        except Exception as e:
            logger.error(f"Error obteniendo datos de reseñas para {app_id}: {e}")
            return {'review_count': 0, 'review_score': 0}

    def get_early_access_data(self, app_id: str) -> Dict:
        """Obtiene datos de Early Access de SteamDB"""
        try:
            url = f"https://steamdb.info/app/{app_id}"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                ea_div = soup.find('div', {'class': 'early-access'})
                
                if ea_div:
                    return {
                        'early_access_date': ea_div.find('span', {'class': 'date-start'}).text,
                        'full_release_date': ea_div.find('span', {'class': 'date-end'}).text
                    }
            return {'early_access_date': None, 'full_release_date': None}
        except Exception as e:
            logger.error(f"Error obteniendo datos de Early Access para {app_id}: {e}")
            return {'early_access_date': None, 'full_release_date': None}

    def verify_data_collection(self, df: pd.DataFrame) -> bool:
        """Verifica que todos los campos necesarios se han recolectado correctamente"""
        logger.info("\nVerificando recolección de datos...")
        
        try:
            # 1. Verificar campos requeridos
            required_fields = {
                'app_id': int,
                'name': str,
                'price': float,
                'is_free': bool,
                'owners': int,
                'genres': str,
                'is_indie': bool,
                'current_players': int,
                'review_count': int,
                'review_score': float,
                'metacritic_score': float,
                'user_score': float,
                'critic_reviews_count': int
            }
            
            missing_fields = []
            wrong_types = []
            
            for field, expected_type in required_fields.items():
                if field not in df.columns:
                    missing_fields.append(field)
                elif not df[field].dtype == expected_type:
                    wrong_types.append(f"{field} (esperado: {expected_type}, actual: {df[field].dtype})")
            
            if missing_fields:
                logger.error(f"Campos faltantes: {missing_fields}")
                return False
                
            if wrong_types:
                logger.error(f"Tipos de datos incorrectos: {wrong_types}")
                return False
            
            # 2. Verificar rangos válidos
            validations = {
                'price': (df['price'] >= 0).all(),
                'owners': (df['owners'] >= 0).all(),
                'current_players': (df['current_players'] >= 0).all(),
                'review_score': ((df['review_score'] >= 0) & (df['review_score'] <= 100)).all(),
                'metacritic_score': ((df['metacritic_score'] >= 0) & (df['metacritic_score'] <= 100)).all(),
                'user_score': ((df['user_score'] >= 0) & (df['user_score'] <= 10)).all()
            }
            
            for field, is_valid in validations.items():
                if not is_valid:
                    logger.error(f"Valores fuera de rango en {field}")
                    return False
                
            # 3. Verificar completitud de datos
            completeness = {
                'Metacritic Score': len(df[df['metacritic_score'] > 0]),
                'Current Players': len(df[df['current_players'] > 0]),
                'Reviews': len(df[df['review_count'] > 0]),
                'User Score': len(df[df['user_score'] > 0])
            }
            
            logger.info("\nCompletitud de datos:")
            for field, count in completeness.items():
                percentage = (count / len(df)) * 100
                logger.info(f"{field}: {count}/{len(df)} ({percentage:.1f}%)")
            
            # 4. Verificar consistencia
            inconsistencies = []
            
            # Verificar consistencia de is_free
            free_price_mismatch = len(df[(df['is_free'] & (df['price'] > 0)) | (~df['is_free'] & (df['price'] == 0))])
            if free_price_mismatch > 0:
                inconsistencies.append(f"Inconsistencia en is_free: {free_price_mismatch} registros")
            
            # Verificar consistencia de reviews
            if len(df[df['review_count'] > 0]) < len(df[df['review_score'] > 0]):
                inconsistencies.append("Inconsistencia en reviews: hay scores sin conteo de reviews")
            
            if inconsistencies:
                logger.warning("Inconsistencias encontradas:")
                for inc in inconsistencies:
                    logger.warning(f"- {inc}")
            
            logger.info("✓ Verificación de datos completada exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error en verificación de datos: {e}")
            logger.exception("Detalles del error:")
            return False

    def get_dlc_data(self, app_id: str) -> Dict:
        """Obtiene información sobre DLCs desde SteamDB"""
        try:
            url = f"https://steamdb.info/search/?a=all&q=dlc"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Buscar DLCs para el juego específico
                dlc_table = soup.find('table', {'class': 'table-products'})
                if dlc_table:
                    dlcs = dlc_table.find_all('tr', {'data-appid': app_id})
                    
                    return {
                        'dlc_count': len(dlcs),
                        'has_dlc': len(dlcs) > 0,
                        'dlc_names': [dlc.find('td', {'class': 'name'}).text.strip() for dlc in dlcs]
                    }
            
            return {'dlc_count': 0, 'has_dlc': False, 'dlc_names': []}
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de DLC para {app_id}: {e}")
            return {'dlc_count': 0, 'has_dlc': False, 'dlc_names': []}

    def get_microtransactions_data(self, app_id: str) -> Dict:
        """Obtiene información sobre microtransacciones desde Steam Store"""
        try:
            url = f"https://store.steampowered.com/app/{app_id}"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Buscar indicadores de microtransacciones
                microtrans_indicators = [
                    'in-app purchases',
                    'microtransactions',
                    'in-game purchases',
                    'virtual currency',
                    'premium currency'
                ]
                
                page_text = soup.get_text().lower()
                has_microtrans = any(indicator in page_text for indicator in microtrans_indicators)
                
                # Buscar elementos específicos que indican microtransacciones
                store_tags = soup.find_all('a', {'class': 'app_tag'})
                tags_text = ' '.join(tag.text.lower() for tag in store_tags if tag)
                
                return {
                    'has_microtransactions': has_microtrans or any(indicator in tags_text for indicator in microtrans_indicators),
                    'microtransaction_type': self._determine_microtrans_type(soup) if has_microtrans else 'None'
                }
            
            return {'has_microtransactions': False, 'microtransaction_type': 'None'}
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de microtransacciones para {app_id}: {e}")
            return {'has_microtransactions': False, 'microtransaction_type': 'None'}

    def _determine_microtrans_type(self, soup) -> str:
        """Determina el tipo de microtransacciones basado en el contenido de la página"""
        try:
            page_text = soup.get_text().lower()
            
            if 'cosmetic' in page_text or 'skins' in page_text:
                return 'Cosmetic'
            elif 'loot box' in page_text or 'gacha' in page_text:
                return 'Loot Boxes'
            elif 'battle pass' in page_text or 'season pass' in page_text:
                return 'Battle Pass'
            elif 'premium currency' in page_text or 'virtual currency' in page_text:
                return 'Virtual Currency'
            else:
                return 'General'
                
        except Exception:
            return 'Unknown'

    def monitor_progress(self, current_count: int, target_total: int, start_time: float, successful_requests: int, errors: int):
        """Monitorea y muestra el progreso de la recolección de datos"""
        try:
            # Calcular tiempo transcurrido y estimaciones
            elapsed_time = time.time() - start_time
            current_rate = current_count / (elapsed_time / 60) if elapsed_time > 0 else 0
            remaining_games = target_total - current_count
            estimated_remaining = remaining_games / current_rate if current_rate > 0 else 0
            
            # Mostrar estadísticas generales
            logger.info("\n=== ESTADO DE LA RECOLECCIÓN ===")
            logger.info(f"Progreso: {current_count}/{target_total} juegos ({(current_count/target_total*100):.1f}%)")
            logger.info(f"Tiempo transcurrido: {elapsed_time/60:.1f} minutos")
            logger.info(f"Velocidad actual: {current_rate:.1f} juegos/minuto")
            logger.info(f"Tiempo restante estimado: {estimated_remaining:.1f} minutos")
            
            # Mostrar estadísticas de éxito/error
            success_rate = (successful_requests/(successful_requests+errors))*100 if (successful_requests+errors) > 0 else 0
            logger.info("\n=== ESTADÍSTICAS DE PETICIONES ===")
            logger.info(f"Peticiones exitosas: {successful_requests}")
            logger.info(f"Errores: {errors}")
            logger.info(f"Tasa de éxito: {success_rate:.1f}%")
            
            # Mostrar barra de progreso
            progress = int((current_count/target_total) * 50)
            bar = "=" * progress + "-" * (50 - progress)
            logger.info(f"\n[{bar}] {(current_count/target_total*100):.1f}%")
            
            # Mostrar advertencias si hay problemas
            if success_rate < 80:
                logger.warning("⚠️ Tasa de éxito baja - Posibles problemas de conexión")
            if current_rate < 1:
                logger.warning("⚠️ Velocidad de recolección baja - Revisar límites de API")
            
        except Exception as e:
            logger.error(f"Error monitoreando progreso: {e}")

    def _show_progress(self, games_data: list, start_time: float, target_total: int, successful: int, errors: int, retries: int):
        """Muestra el progreso detallado de la recolección de datos"""
        try:
            current_count = len(games_data)
            elapsed_time = time.time() - start_time
            
            # Calcular métricas de progreso
            current_rate = current_count / (elapsed_time / 60) if elapsed_time > 0 else 0
            remaining_games = target_total - current_count
            estimated_remaining = remaining_games / current_rate if current_rate > 0 else 0
            completion_percentage = (current_count / target_total) * 100
            
            # Mostrar estadísticas generales
            logger.info("\n=== PROGRESO DE RECOLECCIÓN ===")
            logger.info(f"Juegos procesados: {current_count}/{target_total} ({completion_percentage:.1f}%)")
            logger.info(f"Tiempo transcurrido: {elapsed_time/60:.1f} minutos")
            logger.info(f"Velocidad actual: {current_rate:.1f} juegos/minuto")
            logger.info(f"Tiempo restante estimado: {estimated_remaining:.1f} minutos")
            
            # Mostrar estadísticas de éxito/error
            success_rate = (successful/(successful+errors))*100 if (successful+errors) > 0 else 0
            logger.info("\n=== ESTADÍSTICAS DE PETICIONES ===")
            logger.info(f"Peticiones exitosas: {successful}")
            logger.info(f"Errores: {errors}")
            logger.info(f"Reintentos: {retries}")
            logger.info(f"Tasa de éxito: {success_rate:.1f}%")
            
            # Mostrar barra de progreso
            bar_length = 50
            filled_length = int(completion_percentage / 100 * bar_length)
            bar = "=" * filled_length + "-" * (bar_length - filled_length)
            logger.info(f"\nProgreso: [{bar}] {completion_percentage:.1f}%")
            
            # Mostrar últimos juegos procesados
            if games_data:
                logger.info("\nÚltimos juegos procesados:")
                for game in games_data[-5:]:
                    price_info = '[F2P]' if game['is_free'] else f'[${game["price"]:.2f}]'
                    logger.info(f"- {game['name']} ({game['genres']}) {price_info}")
            
            # Mostrar advertencias si hay problemas
            if success_rate < 80:
                logger.warning("⚠️ Tasa de éxito baja - Posibles problemas de conexión")
            if current_rate < 1:
                logger.warning("⚠️ Velocidad de recolección baja - Revisar límites de API")
            
        except Exception as e:
            logger.error(f"Error mostrando progreso: {e}")
            logger.exception("Detalles del error:")

    def _show_final_summary(self, df: pd.DataFrame, start_time: float, successful: int, errors: int):
        """Muestra un resumen final detallado de la recolección"""
        try:
            elapsed_time = time.time() - start_time
            total_games = len(df)
            
            logger.info("\n====== RESUMEN FINAL DE RECOLECCIÓN ======")
            logger.info(f"Total de juegos recolectados: {total_games}")
            logger.info(f"Tiempo total: {elapsed_time/60:.1f} minutos")
            logger.info(f"Velocidad promedio: {total_games/(elapsed_time/60):.1f} juegos/minuto")
            
            # Estadísticas de datos
            logger.info("\n=== ESTADÍSTICAS DE DATOS ===")
            logger.info(f"Juegos gratuitos: {df['is_free'].sum()} ({(df['is_free'].sum()/total_games*100):.1f}%)")
            logger.info(f"Juegos indie: {df['is_indie'].sum()} ({(df['is_indie'].sum()/total_games*100):.1f}%)")
            logger.info(f"Precio promedio (juegos de pago): ${df[~df['is_free']]['price'].mean():.2f}")
            
            # Distribución de géneros
            logger.info("\n=== DISTRIBUCIÓN DE GÉNEROS ===")
            genre_dist = df['genres'].value_counts()
            for genre, count in genre_dist.items():
                logger.info(f"{genre}: {count} ({(count/total_games*100):.1f}%)")
            
            # Verificación de calidad
            logger.info("\n=== VERIFICACIÓN DE CALIDAD ===")
            null_counts = df.isnull().sum()
            if null_counts.any():
                logger.warning("Valores nulos encontrados:")
                for col in null_counts[null_counts > 0].index:
                    logger.warning(f"- {col}: {null_counts[col]} valores nulos")
            
            # Tasa de éxito final
            success_rate = (successful/(successful+errors))*100 if (successful+errors) > 0 else 0
            logger.info(f"\nTasa de éxito final: {success_rate:.1f}%")
            
            if success_rate < 80:
                logger.warning("⚠️ Tasa de éxito por debajo del umbral recomendado (80%)")
                
        except Exception as e:
            logger.error(f"Error mostrando resumen final: {e}")
            logger.exception("Detalles del error:")

    def create_positive_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea positive_ratio combinando múltiples fuentes"""
        logger.info("Calculando positive_ratio desde múltiples fuentes...")
        
        try:
            # 1. Calcular desde reseñas positivas y negativas
            df['positive_ratio'] = df.apply(
                lambda x: x['positive'] / (x['positive'] + x['negative'])
                if (x['positive'] + x['negative']) > 0
                else None,
                axis=1
            )
            
            # 2. Usar review_score donde no hay reseñas
            mask = df['positive_ratio'].isna()
            df.loc[mask, 'positive_ratio'] = df.loc[mask, 'review_score'] / 100
            
            # 3. Usar metacritic_score como respaldo
            mask = df['positive_ratio'].isna()
            df.loc[mask, 'positive_ratio'] = df.loc[mask, 'metacritic_score'] / 100
            
            # 4. Usar user_score como último recurso
            mask = df['positive_ratio'].isna()
            df.loc[mask, 'positive_ratio'] = df.loc[mask, 'user_score'] / 10  # user_score está en escala 0-10
            
            # 5. Rellenar valores restantes con la media o 0.5 si no hay datos
            if df['positive_ratio'].notna().any():
                mean_ratio = df['positive_ratio'].mean()
            else:
                mean_ratio = 0.5
            df['positive_ratio'] = df['positive_ratio'].fillna(mean_ratio)
            
            # 6. Asegurar rango 0-1
            df['positive_ratio'] = df['positive_ratio'].clip(0, 1)
            
            # Mostrar estadísticas
            logger.info("\nEstadísticas de positive_ratio:")
            logger.info(f"Media: {df['positive_ratio'].mean():.2f}")
            logger.info(f"Mediana: {df['positive_ratio'].median():.2f}")
            logger.info(f"Valores únicos: {len(df['positive_ratio'].unique())}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error creando positive_ratio: {e}")
            logger.exception("Detalles del error:")
            # En caso de error, crear un valor por defecto
            df['positive_ratio'] = 0.5
            return df

def main():
    """Función principal para recolectar datos"""
    logging.basicConfig(level=logging.INFO)
    api = SteamSpyAPI()
    
    try:
        # Recolectar dataset comprensivo con más juegos
        logger.info("Iniciando recolección de datos comprensiva...")
        games = api.collect_comprehensive_data(target_total=1000)  # Aumentado a 1000
        
        if isinstance(games, pd.DataFrame):
            logger.info("\nEstadísticas de la recolección:")
            logger.info(f"Total de juegos recolectados: {len(games)}")
            
            # Mostrar algunos ejemplos
            logger.info("\nEjemplos de juegos recolectados:")
            sample_games = games.head()
            for _, game in sample_games.iterrows():
                logger.info(f"- {game['name']} (ID: {game['app_id']})")
                
        # Limpiar archivos temporales
        api.cleanup_temp_files()
            
    except Exception as e:
        logger.error(f"Error en la recolección: {e}")

if __name__ == "__main__":
    main()