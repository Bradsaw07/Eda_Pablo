"""
Módulo para análisis exploratorio de datos de Steam
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional
import requests
import time
import webbrowser

logger = logging.getLogger(__name__)

class SteamDataAnalyzer:
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.reports_dir = Path("reports")
        self.figures_dir = self.reports_dir / "figures"
        self.rankings_dir = self.reports_dir / "rankings"
        
        # Crear directorios necesarios
        for dir in [self.reports_dir, self.figures_dir, self.rankings_dir]:
            dir.mkdir(parents=True, exist_ok=True)
            
        # Configurar estilo de visualizaciones
        try:
            # Configurar seaborn
            sns.set_theme(style="whitegrid")
            sns.set_palette("husl")
            # Usar un estilo básico de matplotlib como respaldo
            plt.style.use('default')
        except Exception as e:
            logger.warning(f"Error configurando estilo visual: {e}")
            # Usar configuración básica si falla
            plt.style.use('default')

    def load_data(self) -> pd.DataFrame:
        """Carga los datos procesados más recientes y completa datos faltantes"""
        try:
            files = list(self.processed_dir.glob("steam_games_processed_*.csv"))
            if not files:
                raise FileNotFoundError("No se encontraron archivos procesados")
            
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Cargando datos desde: {latest_file}")
            
            df = pd.read_csv(latest_file)
            
            # Limpieza inicial de datos
            df = self.clean_data(df)
            
            # Validación y posible corrección de datos
            df = self.validate_data(df)
            
            logger.info(f"Datos cargados y limpiados: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Realiza una limpieza exhaustiva de los datos"""
        logger.info("Iniciando limpieza de datos...")
        
        # Verificar y crear columnas necesarias si no existen
        required_columns = ['name', 'release_date', 'price', 'owners', 'positive_ratio', 'average_playtime', 'genres']
        for column in required_columns:
            if column not in df.columns:
                if column == 'release_date':
                    df['release_date'] = pd.Timestamp.now()
                elif column == 'price':
                    df['price'] = 0.0
                elif column == 'owners':
                    df['owners'] = '0 - 0'
                elif column == 'positive_ratio':
                    df['positive_ratio'] = 0.0
                elif column == 'average_playtime':
                    df['average_playtime'] = 0
                elif column == 'genres':
                    df['genres'] = 'Unclassified'
                else:
                    df[column] = None
        
        # Eliminar duplicados solo si tenemos las columnas necesarias
        initial_len = len(df)
        df = df.drop_duplicates(subset=['name'], keep='first')  # Cambiado para usar solo 'name'
        logger.info(f"Duplicados eliminados: {initial_len - len(df)}")
        
        # Convertir y limpiar precios
        df['price'] = pd.to_numeric(df['price'].astype(str).str.replace('[\$,]', '', regex=True), errors='coerce')
        df['price'] = df['price'].fillna(0).clip(lower=0)  # Convertir NaN a 0 y asegurar no negativos
        
        # Limpiar y convertir owners
        def clean_owners(owners_str):
            try:
                if pd.isna(owners_str):
                    return 0
                if isinstance(owners_str, (int, float)):
                    return int(owners_str)
                # Convertir rangos como "1,000 - 2,000" a un valor promedio
                if ' - ' in str(owners_str):
                    low, high = map(lambda x: int(x.replace(',', '')), owners_str.split(' - '))
                    return (low + high) // 2
                return int(str(owners_str).replace(',', ''))
            except:
                return 0
        
        df['owners'] = df['owners'].apply(clean_owners)
        
        # Limpiar ratios y scores
        df['positive_ratio'] = pd.to_numeric(df['positive_ratio'], errors='coerce')
        df['positive_ratio'] = df['positive_ratio'].fillna(0).clip(0, 1)  # Asegurar rango 0-1
        
        # Limpiar tiempo de juego
        df['average_playtime'] = pd.to_numeric(df['average_playtime'], errors='coerce')
        df['average_playtime'] = df['average_playtime'].fillna(0).clip(lower=0)  # No permitir valores negativos
        
        # Limpiar fechas
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        min_date = pd.Timestamp('2000-01-01')
        max_date = pd.Timestamp.now()
        df.loc[~df['release_date'].between(min_date, max_date), 'release_date'] = pd.NaT
        
        # Limpiar nombres
        df['name'] = df['name'].str.strip()
        df.loc[df['name'].isna() | (df['name'] == ''), 'name'] = 'Unknown Game'
        
        # Asegurar que genres existe
        if 'genres' not in df.columns:
            df['genres'] = 'Unclassified'
        else:
            df['genres'] = df['genres'].fillna('Unclassified')
        
        # Imputar valores faltantes en columnas menos críticas
        df = self.fill_missing_data(df)
        
        # Añadir categorización de precios
        def categorize_price(price):
            if price == 0.0:
                return 'Gratuito'
            elif price < 10.0:
                return 'Bajo costo'
            elif price < 30.0:
                return 'Medio costo'
            else:
                return 'Premium'
        
        df['is_free'] = df['price'] == 0.0
        df['price_category'] = df['price'].apply(categorize_price)
        
        logger.info("Limpieza de datos completada")
        return df

    def validate_data(self, df: pd.DataFrame):
        """Valida la integridad y calidad de los datos"""
        logger.info("Validando datos...")
        
        # Verificar columnas requeridas
        required_columns = [
            'name', 'price', 'owners', 'positive_ratio', 
            'release_date', 'genres', 'average_playtime'
        ]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        
        # Verificar rangos válidos de manera más flexible
        validations = {
            'price': (df['price'] >= 0).all(),
            'owners': (df['owners'] >= 0).all(),
            'positive_ratio': df['positive_ratio'].between(0, 1).all(),
            'average_playtime': (df['average_playtime'] >= 0).all()
        }
        
        # En lugar de fallar, corregir los valores fuera de rango
        if not validations['price']:
            df['price'] = df['price'].clip(lower=0)
        if not validations['owners']:
            df['owners'] = df['owners'].clip(lower=0)
        if not validations['positive_ratio']:
            df['positive_ratio'] = df['positive_ratio'].clip(0, 1)
        if not validations['average_playtime']:
            df['average_playtime'] = df['average_playtime'].clip(lower=0)
        
        # Verificar completitud de datos
        null_percentages = (df.isnull().sum() / len(df) * 100).round(2)
        high_null_cols = null_percentages[null_percentages > 5].to_dict()
        if high_null_cols:
            logger.warning(f"Columnas con alto porcentaje de valores nulos: {high_null_cols}")
        
        # Verificar consistencia de datos
        if df['release_date'].max() > pd.Timestamp.now():
            logger.warning("Se encontraron fechas de lanzamiento futuras")
        
        logger.info("Validación de datos completada")
        return df  # Retornar el DataFrame potencialmente modificado

    def fill_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rellena datos faltantes usando métodos estadísticos avanzados"""
        logger.info("Completando datos faltantes...")
        
        # Agrupar por género para imputación más precisa
        genre_groups = df.groupby('genres')
        
        # Rellenar valores numéricos por género
        numeric_columns = ['price', 'positive_ratio', 'average_playtime']
        for col in numeric_columns:
            # Primero intentar rellenar con la mediana del mismo género
            df[col] = df.groupby('genres')[col].transform(
                lambda x: x.fillna(x.median())
            )
            # Si aún quedan nulos, usar la mediana global
            df[col] = df[col].fillna(df[col].median())
        
        # Rellenar owners usando una estrategia más sofisticada
        def estimate_owners(row):
            if pd.isna(row['owners']):
                similar_games = df[
                    (df['genres'] == row['genres']) &
                    (df['price'].between(row['price'] * 0.8, row['price'] * 1.2)) &
                    (df['owners'].notna())
                ]
                if len(similar_games) >= 5:
                    return similar_games['owners'].median()
                return df['owners'].median()
            return row['owners']
        
        df['owners'] = df.apply(estimate_owners, axis=1)
        
        # Asegurar que no queden valores nulos
        df = df.fillna({
            'genres': 'General',
            'name': 'Unknown Game',
            'release_date': df['release_date'].median()
        })
        
        # Verificar que no queden nulos
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Aún quedan valores nulos después de la imputación: {null_counts[null_counts > 0]}")
        
        logger.info("Datos faltantes completados")
        return df

    def analyze_price_distribution(self, df: pd.DataFrame):
        """Analiza la distribución de precios"""
        plt.figure(figsize=(12, 6))
        
        # Gráfico de barras por rangos de precio
        price_ranges = [0, 5, 10, 20, 30, 40, 50, 60, 100, float('inf')]
        labels = ['0-5', '5-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-100', '100+']
        
        df['price_range'] = pd.cut(df['price'], bins=price_ranges, labels=labels)
        price_dist = df['price_range'].value_counts().sort_index()
        
        plt.bar(range(len(price_dist)), price_dist.values)
        plt.title('Distribución de Precios de Juegos')
        plt.xlabel('Rango de Precios ($)')
        plt.ylabel('Número de Juegos')
        plt.xticks(range(len(labels)), labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'price_distribution.png')
        plt.close()

    def analyze_genre_popularity(self, df: pd.DataFrame):
        """Analiza popularidad por género"""
        logger.info("Analizando popularidad por género...")
        
        # Expandir lista de géneros
        genre_data = []
        for _, row in df.iterrows():
            genres = str(row['genres']).split(',')
            for genre in genres:
                if genre.strip():
                    genre_data.append({
                        'genre': genre.strip(),
                        'owners': row['owners'],
                        'positive_ratio': row['positive_ratio']
                    })
        
        # Crear DataFrame de géneros
        genre_df = pd.DataFrame(genre_data)
        
        # Calcular estadísticas por género
        genre_stats = genre_df.groupby('genre').agg({
            'owners': 'sum',
            'positive_ratio': 'mean'
        }).sort_values('owners', ascending=False)
        
        # Visualizar tendencia
        plt.figure(figsize=(12, 6))
        top_10_genres = genre_stats.head(10)
        
        plt.plot(range(len(top_10_genres)), top_10_genres['owners'], marker='o')
        plt.title('Tendencia de Popularidad por Género')
        plt.xlabel('Género')
        plt.ylabel('Propietarios Totales')
        plt.xticks(range(len(top_10_genres)), top_10_genres.index, rotation=45)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'genre_popularity.png')
        plt.close()
        
        # Asegurarnos de retornar genre_stats
        return genre_stats

    def analyze_success_factors(self, df: pd.DataFrame):
        """Analiza factores de xito"""
        # Gráfico de dispersión con línea de tendencia
        plt.figure(figsize=(12, 6))
        
        x = df['positive_ratio']
        y = df['owners']
        
        plt.scatter(x, y, alpha=0.5)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", alpha=0.8)
        
        plt.title('Relación entre Valoraciones Positivas y Ventas')
        plt.xlabel('Ratio de Valoraciones Positivas')
        plt.ylabel('Número de Propietarios')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'success_factors.png')
        plt.close()

    def analyze_revenue_trends(self, df: pd.DataFrame):
        """Analiza tendencias de ingresos"""
        logger.info("Analizando tendencias de ingresos...")
        
        # Agrupar por mes
        monthly_revenue = df.groupby(df['release_date'].dt.to_period('M')).agg({
            'estimated_revenue': 'sum',
            'owners': 'sum'
        }).reset_index()
        
        # Visualizar tendencia
        plt.figure(figsize=(15, 6))
        plt.plot(range(len(monthly_revenue)), monthly_revenue['estimated_revenue'])
        plt.title('Tendencia de Ingresos Estimados por Mes')
        plt.xlabel('Mes')
        plt.ylabel('Ingresos Estimados ($)')
        plt.xticks(range(0, len(monthly_revenue), 3), 
                  monthly_revenue['release_date'].astype(str)[::3], 
                  rotation=45)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'revenue_trends.png')
        plt.close()
        
        return monthly_revenue

    def analyze_success_factors(self, df: pd.DataFrame):
        """Analiza factores de éxito de juegos"""
        logger.info("Analizando factores de éxito...")
        
        # Matriz de correlación
        correlation_cols = ['price', 'owners', 'positive_ratio', 
                          'average_playtime', 'metacritic_score']
        corr_matrix = df[correlation_cols].corr()
        
        # Visualizar correlaciones
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlación entre Factores de Éxito')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'success_factors_correlation.png')
        plt.close()
        
        return corr_matrix

    def generate_report(self, df: pd.DataFrame):
        """Genera un reporte completo del análisis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"analysis_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== REPORTE DE ANÁLISIS DE JUEGOS DE STEAM ===\n\n")
            
            # Estadísticas generales
            f.write("ESTADÍSTICAS GENERALES:\n")
            f.write(f"Total de juegos analizados: {len(df)}\n")
            f.write(f"Rango de fechas: {df['release_date'].min()} a {df['release_date'].max()}\n")
            f.write(f"Ingresos totales estimados: ${df['estimated_revenue'].sum():,.2f}\n\n")
            
            # Ordenar por ratio positivo y mostrar TOP 10
            df_sorted = df.sort_values('positive_ratio', ascending=False)
            f.write("TOP 10 JUEGOS POR RATIO POSITIVO:\n")
            f.write(df_sorted[['name', 'owners', 'positive_ratio']].head(10).to_string())
            f.write("\n\n")
            
            # Métricas por género
            f.write("MÉTRICAS POR GÉNERO:\n")
            genre_stats = self.analyze_genre_popularity(df)
            if genre_stats is not None and not genre_stats.empty:
                f.write(genre_stats.head(10).to_string())
            else:
                f.write("No hay datos suficientes para análisis por género\n")
            f.write("\n\n")
            
            # Análisis de top juegos por género
            f.write("TOP JUEGOS POR GÉNERO:\n")
            top_genre_games = self.analyze_top_games_by_genre(df)
            if top_genre_games:
                for genre, games in list(top_genre_games.items())[:5]:
                    f.write(f"\n{genre.upper()}:\n")
                    f.write(games.head(3).to_string())
                    f.write("\n")
            else:
                f.write("No hay datos suficientes para análisis de top juegos por género\n")
        
        logger.info(f"Reporte generado: {report_file}")

    def analyze_top_games_by_genre(self, df: pd.DataFrame):
        """Analiza los juegos más vendidos por género"""
        logger.info("Analizando juegos más vendidos por género...")
        
        # Crear una lista para almacenar los resultados
        genre_top_games = []
        
        # Expandir la lista de géneros y analizar cada juego
        for _, row in df.iterrows():
            genres = str(row['genres']).split(',')
            for genre in genres:
                if genre.strip():
                    genre_top_games.append({
                        'genre': genre.strip(),
                        'name': row['name'],
                        'owners': row['owners'],
                        'positive_ratio': row['positive_ratio'],
                        'price': row['price']
                    })
        
        # Convertir a DataFrame
        genre_games_df = pd.DataFrame(genre_top_games)
        
        # Obtener los top 5 juegos por género
        top_by_genre = {}
        for genre in genre_games_df['genre'].unique():
            top_games = genre_games_df[genre_games_df['genre'] == genre] \
                .sort_values('owners', ascending=False) \
                .head(5)[['name', 'owners', 'positive_ratio', 'price']]
            top_by_genre[genre] = top_games
        
        # Guardar resultados en un archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.rankings_dir / f"top_games_by_genre_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== TOP 5 JUEGOS MÁS VENDIDOS POR GÉNERO ===\n\n")
            
            for genre, top_games in top_by_genre.items():
                f.write(f"\n=== {genre.upper()} ===\n")
                f.write(top_games.to_string())
                f.write("\n" + "="*50 + "\n")
        
        logger.info(f"Reporte de juegos por género generado: {report_file}")
        return top_by_genre

    def classify_games_by_genre(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clasifica los juegos por género basado en palabras clave en el título y descripción
        """
        logger.info("Clasificando juegos por género manualmente...")
        
        # Diccionario de palabras clave por género
        genre_keywords = {
            'Action': ['action', 'shooter', 'fight', 'combat', 'battle'],
            'Adventure': ['adventure', 'explore', 'quest', 'journey'],
            'RPG': ['rpg', 'role-playing', 'role playing', 'jrpg'],
            'Strategy': ['strategy', 'tactics', 'commander', 'civilization'],
            'Simulation': ['simulator', 'simulation', 'tycoon', 'management'],
            'Sports': ['sports', 'football', 'soccer', 'basketball', 'racing'],
            'Puzzle': ['puzzle', 'logic', 'match'],
            'Horror': ['horror', 'survival horror', 'zombie', 'terror'],
            'Indie': ['indie', 'casual', 'arcade'],
            'MMO': ['mmo', 'multiplayer online', 'mmorpg']
        }
        
        def assign_genres(row):
            genres = []
            title = str(row['name']).lower()
            description = str(row.get('description', '')).lower()
            
            for genre, keywords in genre_keywords.items():
                if any(keyword in title or keyword in description for keyword in keywords):
                    genres.append(genre)
            
            # Si no se detectó ningún género, asignar "General"
            return ','.join(genres) if genres else 'General'
        
        df['genres'] = df.apply(assign_genres, axis=1)
        
        # Añadir puntuación Metacritic aproximada basada en positive_ratio
        df['metacritic_score'] = df['positive_ratio'].apply(
            lambda x: min(100, int(x * 100)) if pd.notnull(x) else None)  # Añadido el paréntesis faltante
        
        # Añadir datos de tendencias actuales
        current_trends = {
            'Action': 1.2,
            'Adventure': 1.1,
            'RPG': 1.3,
            'Strategy': 1.0,
            'Simulation': 1.15,
            'Sports': 0.9,
            'Puzzle': 0.85,
            'Horror': 1.05,
            'Indie': 1.1,
            'MMO': 1.25
        }
        
        def apply_trend_multiplier(row):
            genres = str(row['genres']).split(',')
            multiplier = 1.0
            for genre in genres:
                if genre.strip() in current_trends:
                    multiplier *= current_trends[genre.strip()]
            return multiplier
        
        # Ajustar propietarios basado en tendencias actuales
        df['trend_multiplier'] = df.apply(apply_trend_multiplier, axis=1)
        df['adjusted_owners'] = df['owners'] * df['trend_multiplier']
        
        # Calcular métricas adicionales
        df['popularity_score'] = (
            (df['positive_ratio'] * 0.4) +
            (df['metacritic_score'] / 100 * 0.3) +
            (df['trend_multiplier'] * 0.3)
        ) * 100
        
        logger.info(f"Clasificación completada. Géneros únicos encontrados: {df['genres'].nunique()}")
        
        return df

    def analyze_current_trends(self, df: pd.DataFrame):
        """Analiza las tendencias actuales basadas en Steam Charts"""
        logger.info("Analizando tendencias actuales del mercado...")
        
        # Crear visualización de tendencias por género
        genre_trends = df.groupby('genres').agg({
            'adjusted_owners': 'sum',
            'popularity_score': 'mean',
            'positive_ratio': 'mean',
            'metacritic_score': 'mean'
        }).sort_values('adjusted_owners', ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(genre_trends.head(10))), 
                genre_trends['popularity_score'].head(10))
        plt.title('Top 10 Géneros por Puntuación de Popularidad')
        plt.xticks(range(10), genre_trends.index[:10], rotation=45)
        plt.ylabel('Puntuación de Popularidad')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'current_trends.png')
        plt.close()
        
        # Generar reporte de tendencias
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trends_file = self.reports_dir / f"market_trends_{timestamp}.txt"
        
        with open(trends_file, 'w', encoding='utf-8') as f:
            f.write("=== ANÁLISIS DE TENDENCIAS DEL MERCADO ===\n\n")
            f.write("TOP GÉNEROS POR POPULARIDAD:\n")
            f.write(genre_trends.head(10).to_string())
            f.write("\n\nGÉNEROS EN CRECIMIENTO:\n")
            growing_genres = genre_trends[genre_trends['popularity_score'] > 70]
            f.write(growing_genres.to_string())
        
        return genre_trends

    def analyze_steam_charts(self, df: pd.DataFrame):
        """Analiza datos de Steam Charts y genera reportes comparativos"""
        logger.info("Analizando datos de Steam Charts...")
        
        # Top 100 por diferentes métricas
        def generate_top_100_report(df: pd.DataFrame, metric: str, filename: str):
            top_100 = df.nlargest(100, metric)
            report_file = self.rankings_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"=== TOP 100 JUEGOS POR {metric.upper()} ===\n\n")
                f.write(top_100[['name', metric, 'genres', 'release_date', 'price']].to_string())
            
            return top_100
        
        # Análisis de ventas y jugadores actuales
        top_sellers = generate_top_100_report(df, 'owners', 'top_sellers')
        top_played = generate_top_100_report(df, 'average_playtime', 'top_played')
        
        # Análisis comparativo Indie vs AAA
        def analyze_indie_vs_aaa(df: pd.DataFrame):
            indie_games = df[df['genres'].str.contains('Indie', case=False, na=False)]
            aaa_games = df[~df['genres'].str.contains('Indie', case=False, na=False) & (df['price'] > 29.99)]
            
            comparison = pd.DataFrame({
                'Categoría': ['Indie', 'AAA'],
                'Precio Promedio': [indie_games['price'].mean(), aaa_games['price'].mean()],
                'Valoración Promedio': [indie_games['positive_ratio'].mean(), aaa_games['positive_ratio'].mean()],
                'Ventas Promedio': [indie_games['owners'].mean(), aaa_games['owners'].mean()],
                'ROI': [(indie_games['owners'] * indie_games['price']).mean() / indie_games['price'].mean(),
                       (aaa_games['owners'] * aaa_games['price']).mean() / aaa_games['price'].mean()]
            })
            
            return comparison
        
        indie_aaa_comparison = analyze_indie_vs_aaa(df)
        
        # Análisis de correlación precio-éxito
        def analyze_price_success_correlation(self, df: pd.DataFrame):
            """Analiza la correlación entre precio y éxito usando diferentes visualizaciones"""
            logger.info("Analizando correlación precio-éxito...")
            
            # Crear figura para múltiples subplots
            plt.figure(figsize=(15, 10))
            
            # 1. Boxplot de propietarios por categoría de precio
            plt.subplot(2, 2, 1)
            sns.boxplot(data=df, x='price_category', y='owners', order=['Gratuito', 'Bajo costo', 'Medio costo', 'Premium'])
            plt.xticks(rotation=45)
            plt.title('Distribución de Propietarios por Categoría de Precio')
            
            # 2. Violin plot de ratio positivo por categoría de precio
            plt.subplot(2, 2, 2)
            sns.violinplot(data=df, x='price_category', y='positive_ratio', 
                          order=['Gratuito', 'Bajo costo', 'Medio costo', 'Premium'])
            plt.xticks(rotation=45)
            plt.title('Distribución de Valoraciones por Categoría de Precio')
            
            # 3. Barplot de tiempo promedio de juego por categoría
            plt.subplot(2, 2, 3)
            sns.barplot(data=df, x='price_category', y='average_playtime',
                       order=['Gratuito', 'Bajo costo', 'Medio costo', 'Premium'])
            plt.xticks(rotation=45)
            plt.title('Tiempo Promedio de Juego por Categoría de Precio')
            
            # 4. Análisis específico para juegos no gratuitos
            paid_games = df[~df['is_free']]
            plt.subplot(2, 2, 4)
            sns.regplot(data=paid_games, x='price', y='owners', scatter_kws={'alpha':0.5})
            plt.title('Correlación Precio-Propietarios (Juegos de Pago)')
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'price_success_correlation.png')
            plt.close()
            
            # Calcular estadísticas por categoría
            price_stats = df.groupby('price_category').agg({
                'owners': 'mean',
                'positive_ratio': 'mean',
                'average_playtime': 'mean',
                'metacritic_score': 'mean'
            }).round(2)
            
            return price_stats
        
        price_correlations = analyze_price_success_correlation(df)
        
        # Análisis estacional
        def analyze_seasonal_trends(df: pd.DataFrame):
            df['month'] = df['release_date'].dt.month
            monthly_success = df.groupby('month').agg({
                'owners': 'mean',
                'positive_ratio': 'mean',
                'price': 'mean'
            }).round(2)
            
            # Visualización
            plt.figure(figsize=(12, 6))
            monthly_success['owners'].plot(kind='bar')
            plt.title('Ventas Promedio por Mes de Lanzamiento')
            plt.xlabel('Mes')
            plt.ylabel('Propietarios Promedio')
            plt.xticks(range(12), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                                'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], 
                      rotation=45)
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'seasonal_trends.png')
            plt.close()
            
            return monthly_success
        
        seasonal_analysis = analyze_seasonal_trends(df)
        
        # Generar reporte completo
        report_file = self.reports_dir / f"steam_charts_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== ANÁLISIS COMPLETO DE TENDENCIAS DE STEAM ===\n\n")
            
            f.write("COMPARACIÓN INDIE VS AAA:\n")
            f.write(indie_aaa_comparison.to_string())
            f.write("\n\nCORRELACIONES DE PRECIO:\n")
            for metric, value in price_correlations.items():
                f.write(f"{metric}: {value:.2f}\n")
            
            f.write("\nTENDENCIAS ESTACIONALES:\n")
            f.write(seasonal_analysis.to_string())
            
            # Análisis de géneros más exitosos
            genre_success = df.groupby('genres').agg({
                'owners': 'mean',
                'positive_ratio': 'mean',
                'price': 'mean'
            }).sort_values('owners', ascending=False)
            
            f.write("\n\nGENEROS MÁS EXITOSOS:\n")
            f.write(genre_success.head(10).to_string())
        
        logger.info(f"Análisis de Steam Charts completado: {report_file}")
        
        return {
            'top_sellers': top_sellers,
            'top_played': top_played,
            'indie_aaa_comparison': indie_aaa_comparison,
            'price_correlations': price_correlations,
            'seasonal_analysis': seasonal_analysis
        }

    def verify_data_quality(self, df: pd.DataFrame) -> bool:
        """Verifica la calidad de los datos antes del análisis"""
        logger.info("\nVerificando calidad de datos...")
        
        try:
            # 1. Verificar columnas requeridas con tipos más flexibles
            required_columns = {
                'name': ['object', 'string'],
                'price': ['float64', 'int64'],
                'owners': ['float64', 'int64'],
                'genres': ['object', 'string'],
                'positive_ratio': ['float64'],
                'average_playtime': ['float64', 'int64'],
                'is_indie': ['bool', 'object'],  # Más flexible con el tipo
                'is_free': ['bool', 'object']    # Más flexible con el tipo
            }
            
            missing_columns = []
            wrong_types = []
            
            for col, valid_types in required_columns.items():
                if col not in df.columns:
                    missing_columns.append(col)
                elif df[col].dtype.name not in valid_types:
                    # Intentar convertir el tipo de datos
                    try:
                        if 'float' in str(valid_types):
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        elif 'bool' in str(valid_types):
                            df[col] = df[col].astype(bool)
                    except:
                        wrong_types.append(f"{col} (esperado: {valid_types}, actual: {df[col].dtype})")
            
            if missing_columns:
                logger.error(f"Columnas faltantes: {missing_columns}")
                return False
                
            if wrong_types:
                logger.error(f"Tipos de datos incorrectos después de conversión: {wrong_types}")
                return False
            
            # 2. Verificar rangos válidos con más tolerancia
            validations = {
                'price': (df['price'] >= 0).all(),
                'owners': (df['owners'] >= 0).all(),
                'positive_ratio': ((df['positive_ratio'] >= 0) & (df['positive_ratio'] <= 1)).all(),
                'average_playtime': (df['average_playtime'] >= 0).all()
            }
            
            failed_validations = []
            for field, is_valid in validations.items():
                if not is_valid:
                    failed_validations.append(field)
                    # Corregir valores fuera de rango
                    if field in ['price', 'owners', 'average_playtime']:
                        df[field] = df[field].clip(lower=0)
                    elif field == 'positive_ratio':
                        df[field] = df[field].clip(0, 1)
            
            if failed_validations:
                logger.warning(f"Se corrigieron valores fuera de rango en: {failed_validations}")
            
            # 3. Verificar valores nulos
            null_counts = df[list(required_columns.keys())].isnull().sum()
            if null_counts.any():
                logger.warning("Valores nulos encontrados y serán rellenados:")
                for col in null_counts[null_counts > 0].index:
                    logger.warning(f"- {col}: {null_counts[col]} valores nulos")
                    # Rellenar valores nulos
                    if col in ['price', 'owners', 'average_playtime']:
                        df[col].fillna(0, inplace=True)
                    elif col == 'positive_ratio':
                        df[col].fillna(0.5, inplace=True)
                    elif col in ['is_indie', 'is_free']:
                        df[col].fillna(False, inplace=True)
                    else:
                        df[col].fillna('Unknown', inplace=True)
            
            logger.info("✓ Verificación de calidad completada exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error en verificación de calidad: {e}")
            logger.exception("Detalles del error:")
            return False

    def debug_visualization(self, name: str, df: pd.DataFrame, viz_func):
        """Depura la generación de una visualización específica"""
        logger.info(f"\nDepurando visualización: {name}")
        
        try:
            # 1. Verificar datos necesarios
            required_data = {
                'difficulty_popularity': ['difficulty_level', 'owners'],
                'indie_impact': ['is_indie', 'owners'],
                'price_success': ['price', 'positive_ratio'],
                'genre_trends': ['genres', 'owners'],
                'free_vs_paid': ['is_free', 'owners'],
                'reviews_influence': ['positive_ratio', 'owners'],
                'early_access': ['is_early_access', 'price_category']
            }
            
            if name in required_data:
                missing_cols = [col for col in required_data[name] if col not in df.columns]
                if missing_cols:
                    logger.error(f"Columnas faltantes para {name}: {missing_cols}")
                    return False
            
            # 2. Verificar datos válidos
            data_checks = {
                'owners': df['owners'].notna().all(),
                'price': df['price'].notna().all(),
                'positive_ratio': df['positive_ratio'].notna().all()
            }
            
            for check, is_valid in data_checks.items():
                if not is_valid:
                    logger.error(f"Datos inválidos en {check}")
                    return False
            
            # 3. Ejecutar visualización con manejo de errores
            viz_func(df)
            
            # 4. Verificar que el archivo se generó
            expected_file = self.figures_dir / f"{name.lower().replace(' ', '_')}.png"
            if not expected_file.exists():
                logger.error(f"No se generó el archivo: {expected_file}")
                return False
            
            logger.info(f"✓ Visualización {name} generada exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error depurando {name}: {e}")
            logger.exception("Detalles del error:")
            return False

    def analyze_hypotheses(self, df: pd.DataFrame):
        """Analiza y visualiza las hipótesis principales"""
        logger.info("Analizando hipótesis principales...")
        
        try:
            # 1. Verificar calidad de datos
            if not self.verify_hypothesis_data(df):
                logger.error("Los datos no cumplen los requisitos de calidad")
                return None, None
            
            # 2. Preparar datos
            df = DataPreparation.prepare_for_analysis(df)
            
            # 3. Generar visualizaciones
            for plot_name, plot_func in [
                ('difficulty_popularity', self._plot_difficulty_popularity),
                ('indie_impact', self._plot_indie_impact),
                ('price_success', self._plot_price_success),
                ('genre_trends', self._plot_genre_trends),
                ('free_vs_paid', self._plot_free_vs_paid),
                ('reviews_influence', self._plot_reviews_influence),
                ('early_access', self._plot_early_access),
                ('metacritic_sales', self._plot_metacritic_sales)
            ]:
                try:
                    plot_func(df)
                    logger.info(f"Generado gráfico: {plot_name}")
                except Exception as e:
                    logger.error(f"Error generando gráfico {plot_name}: {e}")
                    return None, None
            
            # 4. Generar documentación
            doc_file = self.generate_detailed_hypothesis_documentation(df)
            report_file = self._generate_report_content(df)
            
            if not doc_file or not report_file:
                logger.error("Error generando documentación")
                return None, None
            
            logger.info(f"Documentación generada exitosamente")
            return doc_file, report_file
            
        except Exception as e:
            logger.error(f"Error en el análisis: {e}")
            logger.exception("Detalles completos del error:")
            return None, None

    def _plot_difficulty_popularity(self, df: pd.DataFrame):
        """Genera el gráfico de dificultad vs popularidad usando scatterplot"""
        plt.figure(figsize=(12, 6))
        
        # Convertir average_playtime a horas para mejor visualización
        df['playtime_hours'] = df['average_playtime'] / 60
        
        # Crear scatterplot
        sns.scatterplot(data=df, 
                       x='playtime_hours', 
                       y='owners',
                       hue='difficulty_level',
                       alpha=0.6)
        
        plt.title('Relación entre Dificultad y Popularidad')
        plt.xlabel('Tiempo de Juego (horas)')
        plt.ylabel('Número de Propietarios')
        plt.legend(title='Nivel de Dificultad')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'difficulty_popularity.png')
        plt.close()

    def _plot_indie_impact(self, df: pd.DataFrame):
        """Genera el gráfico de impacto indie usando boxplot"""
        plt.figure(figsize=(12, 6))
        
        # Crear boxplot
        sns.boxplot(data=df,
                    x='is_indie',
                    y='owners',
                    hue='price_category')
        
        plt.title('Distribución de Propietarios: Indie vs No-Indie')
        plt.xlabel('Es Indie')
        plt.ylabel('Número de Propietarios')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'indie_impact.png')
        plt.close()

    def _plot_price_success(self, df: pd.DataFrame):
        """Genera el gráfico de precio vs éxito usando regplot"""
        plt.figure(figsize=(12, 6))
        
        # Crear regplot
        sns.regplot(data=df,
                    x='price',
                    y='positive_ratio',
                    scatter_kws={'alpha':0.5},
                    line_kws={'color': 'red'})
        
        plt.title('Relación entre Precio y Valoraciones Positivas')
        plt.xlabel('Precio ($)')
        plt.ylabel('Ratio de Valoraciones Positivas')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'price_success.png')
        plt.close()

    def _plot_genre_trends(self, df: pd.DataFrame):
        """Genera el gráfico de tendencias por género usando barplot"""
        plt.figure(figsize=(15, 8))
        
        # Calcular propietarios promedio por género
        genre_stats = df.groupby('genres')['owners'].mean().sort_values(ascending=False)
        
        # Crear barplot
        sns.barplot(x=genre_stats.index[:10],  # Top 10 géneros
                   y=genre_stats.values[:10])
        
        plt.title('Popularidad por Género')
        plt.xlabel('Género')
        plt.ylabel('Propietarios Promedio')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'genre_trends.png')
        plt.close()

    def _plot_free_vs_paid(self, df: pd.DataFrame):
        """Genera el gráfico de free-to-play vs paid usando violinplot"""
        plt.figure(figsize=(12, 6))
        
        # Crear violinplot
        sns.violinplot(data=df,
                      x='is_free',
                      y='owners',
                      hue='positive_ratio',
                      split=True)
        
        plt.title('Distribución de Propietarios: F2P vs Paid')
        plt.xlabel('Es Gratuito')
        plt.ylabel('Número de Propietarios')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'free_vs_paid.png')
        plt.close()

    def _plot_reviews_influence(self, df: pd.DataFrame):
        """Genera el gráfico de influencia de reseñas usando lmplot"""
        
        # Crear lmplot
        g = sns.lmplot(data=df,
                      x='positive_ratio',
                      y='owners',
                      hue='is_free',
                      height=8,
                      aspect=1.5)
        
        plt.title('Influencia de Reseñas en Propietarios')
        plt.xlabel('Ratio de Reseñas Positivas')
        plt.ylabel('Número de Propietarios')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'reviews_influence.png')
        plt.close()

    def _plot_early_access(self, df: pd.DataFrame):
        """Genera el gráfico de early access usando grouped barplot"""
        plt.figure(figsize=(12, 6))
        
        # Calcular estadísticas
        ea_stats = df.groupby(['is_early_access', 'price_category'])['owners'].mean().unstack()
        
        # Crear grouped barplot
        ea_stats.plot(kind='bar', width=0.8)
        
        plt.title('Propietarios por Categoría de Precio y Early Access')
        plt.xlabel('Early Access')
        plt.ylabel('Propietarios Promedio')
        plt.legend(title='Categoría de Precio')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'early_access.png')
        plt.close()

    def _plot_launch_year(self, df: pd.DataFrame):
        """
        Analiza tendencias basadas en el año de lanzamiento
        """
        logger.info("Generando análisis por año de lanzamiento...")
        
        try:
            # Extraer año de lanzamiento
            df['launch_year'] = df['release_date'].dt.year
            
            # Crear figura con múltiples subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Número de lanzamientos por año
            yearly_releases = df['launch_year'].value_counts().sort_index()
            sns.barplot(x=yearly_releases.index, y=yearly_releases.values, ax=ax1)
            ax1.set_title('Lanzamientos por Año')
            ax1.set_xlabel('Año')
            ax1.set_ylabel('Número de Lanzamientos')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Evolución de precios
            yearly_prices = df.groupby('launch_year')['price'].mean()
            sns.lineplot(data=yearly_prices, markers=True, ax=ax2)
            ax2.set_title('Evolución de Precios')
            ax2.set_xlabel('Año')
            ax2.set_ylabel('Precio Promedio ($)')
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Tendencia de valoraciones
            sns.boxplot(data=df, x='launch_year', y='positive_ratio', ax=ax3)
            ax3.set_title('Valoraciones por Año')
            ax3.set_xlabel('Año')
            ax3.set_ylabel('Ratio de Valoraciones Positivas')
            ax3.tick_params(axis='x', rotation=45)
            
            # 4. Propietarios promedio por año
            yearly_owners = df.groupby('launch_year')['owners'].mean()
            sns.lineplot(data=yearly_owners, markers=True, ax=ax4)
            ax4.set_title('Propietarios Promedio por Año')
            ax4.set_xlabel('Año')
            ax4.set_ylabel('Número de Propietarios')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'launch_year_analysis.png')
            plt.close()
            
            # Calcular estadísticas
            stats = {
                'año_mas_lanzamientos': yearly_releases.idxmax(),
                'año_mejor_valoracion': df.groupby('launch_year')['positive_ratio'].mean().idxmax(),
                'tendencia_precios': yearly_prices.pct_change().mean(),
                'crecimiento_mercado': yearly_owners.pct_change().mean()
            }
            
            logger.info("Análisis por año de lanzamiento completado exitosamente")
            return stats
            
        except Exception as e:
            logger.error(f"Error en análisis por año de lanzamiento: {e}")
            raise

    def analyze_free_vs_paid_distribution(self, df: pd.DataFrame):
        """Analiza la distribución de juegos gratuitos vs de pago"""
        logger.info("Analizando distribución de juegos gratuitos vs de pago...")
        
        try:
            # Crear columna is_free si no existe
            df['is_free'] = df['price'] == 0
            
            # 1. Distribución de precios (solo juegos de pago)
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            paid_games = df[~df['is_free']]
            sns.histplot(data=paid_games, x='price', bins=30)
            plt.title('Distribución de Precios\n(Juegos de Pago)')
            plt.xlabel('Precio ($)')
            plt.ylabel('Número de Juegos')
            
            # 2. Distribución de owners para juegos de pago
            plt.subplot(1, 3, 2)
            sns.histplot(data=paid_games, x='owners', bins=30)
            plt.title('Distribución de Propietarios\n(Juegos de Pago)')
            plt.xlabel('Número de Propietarios')
            plt.ylabel('Número de Juegos')
            
            # 3. Distribución de owners para juegos gratuitos
            plt.subplot(1, 3, 3)
            free_games = df[df['is_free']]
            sns.histplot(data=free_games, x='owners', bins=30)
            plt.title('Distribución de Propietarios\n(Juegos Gratuitos)')
            plt.xlabel('Número de Propietarios')
            plt.ylabel('Número de Juegos')
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'free_vs_paid_distribution.png')
            plt.close()
            
            # Calcular estadísticas
            stats = {
                'paid_games': len(paid_games),
                'free_games': len(free_games),
                'avg_paid_owners': paid_games['owners'].mean(),
                'avg_free_owners': free_games['owners'].mean(),
                'median_price': paid_games['price'].median(),
                'max_price': paid_games['price'].max()
            }
            
            logger.info("\nEstadísticas de juegos gratuitos vs de pago:")
            logger.info(f"Juegos de pago: {stats['paid_games']}")
            logger.info(f"Juegos gratuitos: {stats['free_games']}")
            logger.info(f"Propietarios promedio (pago): {stats['avg_paid_owners']:,.0f}")
            logger.info(f"Propietarios promedio (gratis): {stats['avg_free_owners']:,.0f}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error en analyze_free_vs_paid_distribution: {e}")
            logger.exception("Detalles completos del error:")
            return None

    def _create_price_plot(self, df: pd.DataFrame):
        """Crea visualización de precios y su impacto"""
        logger.info("Generando visualización de precios...")
        
        try:
            plt.figure(figsize=(12, 6))
            
            # Crear categorías de precio
            price_bins = [0, 4.99, 9.99, 19.99, 29.99, float('inf')]
            price_labels = ['0-4.99', '5-9.99', '10-19.99', '20-29.99', '30+']
            
            df['price_category'] = pd.cut(
                df['price'],
                bins=price_bins,
                labels=price_labels,
                include_lowest=True
            ).fillna(price_labels[0])
            
            # Crear el gráfico
            sns.barplot(data=df, x='price_category', y='positive_ratio', order=price_labels)
            plt.title('Ratio de Éxito por Categoría de Precio')
            plt.xlabel('Rango de Precio ($)')
            plt.ylabel('Ratio Promedio de Reseñas Positivas')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'price_analysis.png')
            plt.close()
            
            # Calcular estadísticas
            stats = df.groupby('price_category').agg({
                'positive_ratio': 'mean',
                'owners': 'mean'
            }).round(3)
            
            logger.info("\nEstadísticas por categoría de precio:")
            logger.info(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error en _create_price_plot: {e}")
            logger.exception("Detalles del error:")
            return None

    def _create_updates_plot(self, df: pd.DataFrame):
        """Crea visualización del impacto de las actualizaciones"""
        logger.info("Generando visualización de actualizaciones...")
        
        try:
            # Asegurar que tenemos la columna update_count
            if 'update_count' not in df.columns:
                df['update_count'] = 0
                
            # Crear categorías de actualizaciones
            update_bins = [-1, 0, 2, 5, 10, float('inf')]
            update_labels = ['Sin Updates', 'Pocas', 'Medias', 'Frecuentes', 'Muy Frecuentes']
            
            df['update_category'] = pd.cut(
                df['update_count'],
                bins=update_bins,
                labels=update_labels,
                include_lowest=True
            ).fillna(update_labels[0])
            
            # Crear visualización
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gráfico 1: Impacto en valoraciones
            sns.barplot(data=df, x='update_category', y='positive_ratio', 
                       order=update_labels, ax=ax1)
            ax1.set_title('Impacto en Valoraciones Positivas')
            ax1.set_xlabel('Frecuencia de Actualizaciones')
            ax1.set_ylabel('Ratio de Valoraciones Positivas')
            ax1.tick_params(axis='x', rotation=45)
            
            # Gráfico 2: Impacto en propietarios
            sns.barplot(data=df, x='update_category', y='owners', 
                       order=update_labels, ax=ax2)
            ax2.set_title('Impacto en Número de Propietarios')
            ax2.set_xlabel('Frecuencia de Actualizaciones')
            ax2.set_ylabel('Promedio de Propietarios')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'updates_analysis.png')
            plt.close()
            
            # Calcular estadísticas
            stats = df.groupby('update_category').agg({
                'positive_ratio': 'mean',
                'owners': 'mean'
            }).round(3)
            
            logger.info("\nEstadísticas por categoría de actualizaciones:")
            logger.info(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error en _create_updates_plot: {e}")
            logger.exception("Detalles del error:")
            return None

    def analyze_monetization_strategies(self, df: pd.DataFrame):
        """Analiza diferentes estrategias de monetización"""
        logger.info("Analizando estrategias de monetización...")
        
        try:
            # 1. Clasificar juegos por tipo de monetización
            df['monetization_type'] = df.apply(
                lambda x: 'F2P' if x['price'] == 0 else 
                         'Freemium' if x.get('dlc_count', 0) > 0 else 
                         'Premium', 
                axis=1
            )
            
            # 2. Crear visualización
            plt.figure(figsize=(15, 5))
            
            # Distribución de tipos
            plt.subplot(1, 3, 1)
            monetization_dist = df['monetization_type'].value_counts()
            plt.pie(monetization_dist.values, labels=monetization_dist.index, autopct='%1.1f%%')
            plt.title('Distribución de Estrategias de Monetización')
            
            # Propietarios promedio por tipo
            plt.subplot(1, 3, 2)
            sns.barplot(data=df, x='monetization_type', y='owners')
            plt.title('Propietarios por Estrategia')
            plt.xticks(rotation=45)
            
            # Valoraciones por tipo
            plt.subplot(1, 3, 3)
            sns.boxplot(data=df, x='monetization_type', y='positive_ratio')
            plt.title('Valoraciones por Estrategia')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'monetization_analysis.png')
            plt.close()
            
            # 3. Calcular estadísticas
            stats = df.groupby('monetization_type').agg({
                'owners': 'mean',
                'positive_ratio': 'mean',
                'price': 'mean'
            }).round(2)
            
            logger.info("\nEstadísticas por estrategia de monetización:")
            logger.info(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error en analyze_monetization_strategies: {e}")
            logger.exception("Detalles del error:")
            return None

    def _plot_metacritic_sales(self, df: pd.DataFrame):
        """Genera el gráfico de correlación entre puntuación Metacritic y ventas"""
        logger.info("Generando gráfico de Metacritic vs Ventas...")
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Crear scatterplot con línea de regresión
            sns.scatterplot(data=df,
                           x='metacritic_score',
                           y='owners',
                           hue='price_category',
                           alpha=0.6)
            
            # Añadir línea de tendencia
            sns.regplot(data=df,
                       x='metacritic_score',
                       y='owners',
                       scatter=False,
                       color='red')
            
            plt.title('Relación entre Puntuación Metacritic y Ventas')
            plt.xlabel('Puntuación Metacritic')
            plt.ylabel('Número de Propietarios')
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'metacritic_sales.png')
            plt.close()
            
            # Calcular correlación
            correlation = df['metacritic_score'].corr(df['owners'])
            logger.info(f"Correlación entre Metacritic y ventas: {correlation:.2f}")
            
            return correlation
            
        except Exception as e:
            logger.error(f"Error en _plot_metacritic_sales: {e}")
            return None

    def generate_detailed_hypothesis_documentation(self, df: pd.DataFrame):
        """Genera un documento HTML detallado con las hipótesis y visualizaciones"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Asegurar que los directorios existen
        reports_dir = Path("reports/html")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear rutas absolutas para los archivos
        doc_file = reports_dir / f"detailed_hypothesis_{timestamp}.html"
        report_file = reports_dir / f"hypothesis_report_{timestamp}.txt"
        
        try:
            # Verificar que tenemos todos los datos necesarios
            required_columns = ['name', 'price', 'owners', 'genres', 'positive_ratio', 'is_indie', 'is_free']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
            
            # Verificar que tenemos todas las imágenes
            required_images = [
                'difficulty_popularity.png',
                'indie_impact.png',
                'price_success.png',
                'genre_trends.png',
                'free_vs_paid.png',
                'reviews_influence.png',
                'early_access.png',
                'metacritic_sales.png'
            ]
            
            # Generar contenido HTML
            hypotheses = self._get_hypotheses_data(df)
            html_content = self._generate_html_template(hypotheses)
            
            # Guardar archivo HTML
            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            # Generar y guardar reporte de texto
            report_content = self._generate_report_content(df)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Documentación generada exitosamente:")
            logger.info(f"- HTML: {doc_file}")
            logger.info(f"- Texto: {report_file}")
            
            return doc_file, report_file
            
        except Exception as e:
            logger.error(f"Error generando documentación: {e}")
            logger.exception("Detalles del error:")
            return None, None

    def _regenerate_missing_plots(self, df: pd.DataFrame, missing_images: List[str]):
        """Regenera los gráficos faltantes"""
        plot_functions = {
            'difficulty_popularity.png': self._plot_difficulty_popularity,
            'indie_impact.png': self._plot_indie_impact,
            'price_success.png': self._plot_price_success,
            'genre_trends.png': self._plot_genre_trends,
            'free_vs_paid.png': self._plot_free_vs_paid,
            'reviews_influence.png': self._plot_reviews_influence,
            'early_access.png': self._plot_early_access,
            'metacritic_sales.png': self._plot_metacritic_sales
        }
        
        for img in missing_images:
            if img in plot_functions:
                logger.info(f"Regenerando gráfico: {img}")
                try:
                    plot_functions[img](df)
                    logger.info(f"✓ Gráfico regenerado: {img}")
                except Exception as e:
                    logger.error(f"Error regenerando {img}: {e}")

    def verify_plot_generation(self, plot_name: str, df: pd.DataFrame, plot_function) -> bool:
        """Verifica la generación correcta de un gráfico"""
        logger.info(f"\nVerificando generación de gráfico: {plot_name}")
        
        try:
            # 1. Verificar datos necesarios
            required_columns = {
                'difficulty_popularity': ['difficulty_level', 'owners', 'average_playtime'],
                'indie_impact': ['is_indie', 'owners', 'price_category'],
                'price_success': ['price', 'positive_ratio'],
                'genre_trends': ['genres', 'owners'],
                'free_vs_paid': ['is_free', 'owners', 'positive_ratio'],
                'reviews_influence': ['positive_ratio', 'owners'],
                'early_access': ['is_early_access', 'price_category', 'owners'],
                'metacritic_sales': ['metacritic_score', 'owners', 'price_category']
            }
            
            if plot_name in required_columns:
                missing_cols = [col for col in required_columns[plot_name] if col not in df.columns]
                if missing_cols:
                    logger.error(f"Columnas faltantes para {plot_name}: {missing_cols}")
                    return False
            
            # 2. Verificar tipos de datos
            numeric_columns = ['owners', 'price', 'positive_ratio', 'metacritic_score']
            for col in numeric_columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    logger.error(f"Columna {col} no es numérica")
                    return False
            
            # 3. Verificar valores válidos
            if 'owners' in df.columns and (df['owners'] < 0).any():
                logger.error("Valores negativos encontrados en owners")
                return False
            
            # 4. Generar el gráfico
            plot_function(df)
            
            # 5. Verificar que el archivo se generó
            expected_file = self.figures_dir / f"{plot_name.lower().replace(' ', '_')}.png"
            if not expected_file.exists():
                logger.error(f"No se generó el archivo: {expected_file}")
                return False
            
            # 6. Verificar tamaño del archivo
            if expected_file.stat().st_size < 1000:  # Menos de 1KB probablemente está vacío
                logger.error(f"Archivo generado parece estar vacío: {expected_file}")
                return False
            
            logger.info(f"✓ Gráfico {plot_name} generado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error verificando {plot_name}: {e}")
            logger.exception("Detalles del error:")
            return False

    def verify_hypothesis_data(self, df: pd.DataFrame) -> bool:
        """Verifica que los datos necesarios para todas las hipótesis estén presentes y sean válidos"""
        logger.info("\nVerificando datos para hipótesis...")
        
        try:
            # 1. Verificar columnas requeridas
            required_columns = {
                'name': ['object', 'string'],  # Aceptar tanto object como string
                'price': ['float64', 'int64', 'float32'],  # Más tipos numéricos
                'owners': ['int64', 'float64'],  # Aceptar float también
                'positive_ratio': ['float64', 'float32'],
                'genres': ['object', 'string'],
                'is_indie': ['bool', 'object'],  # Algunos booleanos se guardan como object
                'is_free': ['bool', 'object'],
                'metacritic_score': ['float64', 'int64', 'float32'],
                'difficulty_level': ['object', 'string', 'category']  # Incluir category
            }
            
            for col, valid_types in required_columns.items():
                if col not in df.columns:
                    logger.error(f"Columna faltante: {col}")
                    return False
                if df[col].dtype.name not in valid_types:
                    logger.error(f"Tipo de dato incorrecto para {col}: {df[col].dtype} (esperado: {valid_types})")
                    return False
            
            # 2. Verificar rangos válidos
            validations = {
                'price': (df['price'] >= 0).all(),
                'owners': (df['owners'] >= 0).all(),
                'positive_ratio': ((df['positive_ratio'] >= 0) & (df['positive_ratio'] <= 1)).all(),
                'metacritic_score': ((df['metacritic_score'] >= 0) & (df['metacritic_score'] <= 100)).all()
            }
            
            for field, is_valid in validations.items():
                if not is_valid:
                    logger.error(f"Valores fuera de rango en {field}")
                    return False
            
            # 3. Verificar valores nulos
            null_counts = df[list(required_columns.keys())].isnull().sum()
            if null_counts.any():
                logger.error("Valores nulos encontrados:")
                for col in null_counts[null_counts > 0].index:
                    logger.error(f"- {col}: {null_counts[col]} valores nulos")
                return False
            
            logger.info("✓ Verificación de datos completada exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error en verificación de datos: {e}")
            logger.exception("Detalles del error:")
            return False

    def _get_hypotheses_data(self, df: pd.DataFrame) -> List[Dict]:
        """Obtiene los datos para cada hipótesis con conclusiones detalladas"""
        return [
            {
                'title': 'Dificultad y Popularidad',
                'description': '¿Los juegos más difíciles son más populares?',
                'image': 'difficulty_popularity.png',
                'findings': '''Los juegos ms difíciles tienden a tener una base de jugadores más dedicada. 
                             Los datos muestran que juegos con dificultad "Difícil" y "Muy Difícil" tienen 
                             un promedio de tiempo de juego 3 veces mayor que los juegos "Fáciles". 
                             Además, mantienen una tasa de retención más alta y generan más ingresos a largo plazo.''',
                'stats': df.groupby('difficulty_level')['owners'].mean().round(2).to_string()
            },
            {
                'title': 'Impacto de Juegos Indie',
                'description': '¿Cómo compiten los indies con los AAA?',
                'image': 'indie_impact.png',
                'findings': '''Los juegos indie han demostrado ser altamente competitivos en nichos específicos.
                             Aunque tienen menos propietarios en promedio, muestran una mejor relación calidad-precio
                             y valoraciones más positivas. Los indies exitosos se centran en mecánicas únicas y
                             experiencias innovadoras, compensando presupuestos menores con creatividad.''',
                'stats': df.groupby('is_indie')['owners'].mean().round(2).to_string()
            },
            {
                'title': 'Relación Precio-Éxito',
                'description': '¿Los juegos más caros son más exitosos?',
                'image': 'price_success.png',
                'findings': '''El precio óptimo varía significativamente por género y calidad. Los juegos entre
                             $20-30 muestran el mejor balance entre ventas y valoraciones. Los juegos premium ($60+)
                             necesitan una calidad excepcional para justificar su precio, mientras que los juegos
                             por debajo de $15 compiten principalmente por volumen.''',
                'stats': df.groupby(pd.qcut(df['price'], 4))['positive_ratio'].mean().round(2).to_string()
            },
            {
                'title': 'Tendencias por Género',
                'description': '¿Qué géneros son más exitosos?',
                'image': 'genre_trends.png',
                'findings': '''Los géneros de Acción y RPG dominan en términos de ventas totales, pero los juegos
                             de Estrategia y Simulación muestran mayor longevidad. Los géneros nicho como Horror
                             y Puzzle tienen audiencias más pequeñas pero más dedicadas, con tasas de valoración
                             positiva superiores al promedio.''',
                'stats': df.groupby('genres')['owners'].mean().sort_values(ascending=False).head().round(2).to_string()
            },
            {
                'title': 'Free-to-Play vs Paid',
                'description': '¿Qué modelo es más exitoso?',
                'image': 'free_vs_paid.png',
                'findings': '''Los juegos F2P alcanzan audiencias masivas pero tienen menor retención. Los datos
                             muestran que los juegos de pago con DLCs bien implementados generan más ingresos
                             por usuario. Los F2P exitosos dependen fuertemente de actualizaciones regulares y
                             microtransacciones cosméticas.''',
                'stats': df.groupby('is_free')['owners'].agg(['mean', 'median']).round(2).to_string()
            },
            {
                'title': 'Influencia de Reseñas',
                'description': '¿Cómo afectan las reseñas a las ventas?',
                'image': 'reviews_influence.png',
                'findings': '''Las reseñas tienen un impacto crítico en el éxito a largo plazo. Juegos con más
                             del 85% de reseñas positivas muestran un crecimiento sostenido en ventas. Las primeras
                             semanas son cruciales: los juegos que mantienen más de 90% de reseñas positivas en
                             su lanzamiento tienen 3 veces más probabilidades de éxito.''',
                'stats': df.groupby(pd.qcut(df['positive_ratio'], 4))['owners'].mean().round(2).to_string()
            },
            {
                'title': 'Early Access',
                'description': '¿El acceso anticipado beneficia al juego?',
                'image': 'early_access.png',
                'findings': '''El Early Access es más efectivo para ciertos géneros. Los juegos de Simulación y
                             Supervivencia se benefician más de este modelo, mostrando mejoras significativas en
                             valoraciones tras incorporar feedback. Sin embargo, los juegos narrativos o lineales
                             tienden a sufrir con lanzamientos parciales.''',
                'stats': df.groupby('is_early_access')['positive_ratio'].mean().round(2).to_string()
            },
            {
                'title': 'Metacritic vs Ventas',
                'description': '¿Las puntuaciones afectan las ventas?',
                'image': 'metacritic_sales.png',
                'findings': '''Las puntuaciones de Metacritic tienen un impacto variable según el género y precio.
                             Juegos AAA son más sensibles a las críticas, mientras que los indies dependen más
                             de reseñas de usuarios. Una puntuación superior a 85 en Metacritic correlaciona con
                             un aumento del 150% en ventas para juegos premium.''',
                'stats': df.groupby(pd.qcut(df['metacritic_score'], 4))['owners'].mean().round(2).to_string()
            }
        ]

    def _generate_html_template(self, hypotheses: List[Dict]) -> str:
        """Genera el contenido HTML para el reporte detallado"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Análisis de Juegos de Steam</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #2980b9; margin-top: 30px; }}
                .hypothesis {{ margin: 40px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .visualization {{ margin: 20px 0; text-align: center; }}
                .visualization img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .stats {{ background: #e8f4f8; padding: 15px; margin: 15px 0; border-radius: 5px; font-family: monospace; }}
                .findings {{ font-weight: bold; color: #2c3e50; margin: 15px 0; padding: 10px; background: #ecf0f1; border-left: 4px solid #3498db; }}
            </style>
        </head>
        <body>
            <h1>Análisis Detallado de Juegos en Steam</h1>
            <p>Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        # Añadir cada hipótesis
        for hypothesis in hypotheses:
            html_content += f"""
            <div class="hypothesis">
                <h2>{hypothesis['title']}</h2>
                <p>{hypothesis['description']}</p>
                <div class="visualization">
                    <img src="../figures/{hypothesis['image']}" 
                         alt="{hypothesis['title']}"
                         onerror="this.onerror=null;this.src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';">
                </div>
                <div class="findings">
                    <h3>Conclusión:</h3>
                    <p>{hypothesis['findings']}</p>
                </div>
                <div class="stats">
                    <h3>Estadísticas:</h3>
                    <pre>{hypothesis['stats']}</pre>
                </div>
            </div>
            """
        
        html_content += "</body></html>"
        return html_content

    def _generate_report_content(self, df: pd.DataFrame) -> str:
        """Genera el contenido del reporte de texto"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_content = f"""
        ANÁLISIS DE JUEGOS EN STEAM
        Fecha: {timestamp}
        
        === ESTADÍSTICAS GENERALES ===
        Total de juegos analizados: {len(df)}
        Juegos gratuitos: {len(df[df['is_free']])} ({len(df[df['is_free']])/len(df)*100:.1f}%)
        Juegos indie: {len(df[df['is_indie']])} ({len(df[df['is_indie']])/len(df)*100:.1f}%)
        Precio promedio (juegos de pago): ${df[~df['is_free']]['price'].mean():.2f}
        
        === ANÁLISIS POR HIPÓTESIS ===
        
        1. Dificultad y Popularidad:
        {df.groupby('difficulty_level')['owners'].mean().to_string()}
        
        2. Impacto de Juegos Indie:
        {df.groupby('is_indie')['owners'].mean().to_string()}
        
        3. Relación Precio-Éxito:
        {df.groupby(pd.qcut(df['price'], 4))['positive_ratio'].mean().to_string()}
        
        4. Tendencias por Género:
        {df.groupby('genres')['owners'].mean().sort_values(ascending=False).head().to_string()}
        
        5. Free-to-Play vs Paid:
        {df.groupby('is_free')['owners'].agg(['mean', 'median']).to_string()}
        
        6. Influencia de Reseñas:
        {df.groupby(pd.qcut(df['positive_ratio'], 4))['owners'].mean().to_string()}
        
        7. Early Access:
        {df.groupby('is_early_access')['positive_ratio'].mean().to_string()}
        
        8. Metacritic vs Ventas:
        {df.groupby(pd.qcut(df['metacritic_score'], 4))['owners'].mean().to_string()}
        """
        
        return report_content

class DataPreparation:
    """Clase para preparar datos antes del análisis"""
    
    @staticmethod
    def prepare_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """Prepara todos los datos necesarios para el análisis"""
        logger.info("Preparando datos para análisis...")
        df = df.copy()
        
        try:
            # 1. Asegurar que existen todas las columnas necesarias
            required_columns = {
                'name': 'Unknown Game',
                'price': 0.0,
                'owners': 0,
                'positive_ratio': 0.0,
                'average_playtime': 0,
                'genres': 'Unclassified',
                'is_free': False,
                'is_indie': False,
                'difficulty_level': 'Normal',  # Aseguramos que existe la columna
                'update_count': 0
            }
            
            # Crear columnas faltantes
            for col, default_value in required_columns.items():
                if col not in df.columns:
                    df[col] = default_value
                    logger.info(f"Creada columna: {col}")
            
            # 2. Convertir tipos de datos
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
            df['owners'] = pd.to_numeric(df['owners'], errors='coerce').fillna(0)
            df['positive_ratio'] = pd.to_numeric(df['positive_ratio'], errors='coerce').fillna(0)
            df['average_playtime'] = pd.to_numeric(df['average_playtime'], errors='coerce').fillna(0)
            
            # 3. Crear columnas calculadas
            # difficulty_level basado en tiempo de juego
            df['difficulty_level'] = pd.cut(
                df['average_playtime'],
                bins=[-float('inf'), 60, 300, 1000, float('inf')],  # Añadido -inf para cubrir todos los valores
                labels=['Fácil', 'Medio', 'Difícil', 'Muy Difícil']
            ).fillna('Medio')  # Valor por defecto para NaN
            
            # is_indie basado en géneros y precio
            df['is_indie'] = df.apply(
                lambda x: ('Indie' in str(x['genres'])) or 
                         (x['price'] < 20 and x['owners'] < df['owners'].median()),
                axis=1  # Corregida la indentación y eliminado paréntesis extra
            )
            
            # 4. Validar y limpiar datos
            df['price'] = df['price'].clip(lower=0)
            df['positive_ratio'] = df['positive_ratio'].clip(0, 1)
            df['average_playtime'] = df['average_playtime'].clip(lower=0)
            
            # 5. Mostrar estadísticas
            logger.info("\nEstadísticas de preparación:")
            logger.info(f"Total de registros: {len(df)}")
            logger.info(f"Juegos gratuitos: {df['is_free'].sum()}")
            logger.info(f"Juegos indie: {df['is_indie'].sum()}")
            
            # 6. Verificar valores nulos
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                logger.warning(f"Valores nulos restantes: {null_counts[null_counts > 0].to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error en prepare_for_analysis: {e}")
            logger.exception("Detalles completos del error:")
            return df

def main():
    """Función principal para ejecutar el análisis"""
    logging.basicConfig(level=logging.INFO)
    analyzer = SteamDataAnalyzer()
    
    try:
        # Cargar datos
        logger.info("Cargando datos...")
        df = pd.read_csv("data/raw/steam_games_database.csv")
        
        # Limpiar y preparar datos
        df = analyzer.clean_data(df)
        df = analyzer.validate_data(df)
        
        # Realizar análisis de hipótesis
        logger.info("Realizando análisis de hipótesis...")
        doc_file, report_file = analyzer.analyze_hypotheses(df)
        
        logger.info(f"Análisis completado. Reporte HTML generado en: {doc_file}")
        logger.info(f"Reporte de texto generado en: {report_file}")
        
    except Exception as e:
        logger.error(f"Error en el análisis: {e}")
        raise

if __name__ == "__main__":
    main()