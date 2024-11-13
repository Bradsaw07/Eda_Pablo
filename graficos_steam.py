import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Cargar datos
df = pd.read_csv('steam_games_database.csv')

# Preprocesamiento de datos
def preprocesar_datos(df):
    # Crear ratio de valoraciones positivas
    df['rating_ratio'] = df['positive'] / (df['positive'] + df['negative'])
    
    # Convertir early_access a booleano
    df['has_early_access'] = df['early_access_date'].notna()
    
    return df

# Función para crear directorio
def crear_directorio():
    import os
    if not os.path.exists('graficos'):
        os.makedirs('graficos')

# 1. Dificultad y Popularidad (versión simple)
def grafico_dificultad_popularidad_simple(df):
    # Definir los intervalos de tiempo (en minutos)
    intervalos = [0, 120, 300, 600, 1200, 2400, float('inf')]
    etiquetas = ['0-2h', '2-5h', '5-10h', '10-20h', '20-40h', '>40h']
    
    # Crear una función para categorizar los tiempos
    def categorizar_tiempo(tiempo):
        for i, limite in enumerate(intervalos[1:]):
            if tiempo <= limite:
                return etiquetas[i]
        return etiquetas[-1]
    
    # Filtrar y preparar datos
    df_filtered = df[df['median_playtime'] > 0].copy()
    df_filtered['intervalo_tiempo'] = df_filtered['median_playtime'].apply(categorizar_tiempo)
    
    # Calcular estadísticas por intervalo
    stats = df_filtered.groupby('intervalo_tiempo').agg({
        'owners': 'mean',
        'positive_ratio': 'mean',
        'positive': 'mean'
    }).reindex(etiquetas).fillna(0)
    
    # Crear figura con dos ejes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Barras para propietarios (eje y izquierdo)
    bars = ax1.bar(
        range(len(etiquetas)),
        stats['owners'],
        color='skyblue',
        alpha=0.7,
        width=0.6
    )
    
    # Línea para ratio de valoraciones positivas (eje y derecho)
    ax2 = ax1.twinx()
    line = ax2.plot(
        range(len(etiquetas)),
        stats['positive_ratio'],
        'ro-',
        linewidth=2,
        markersize=8,
        label='Ratio de valoraciones positivas'
    )
    
    # Configurar ejes y etiquetas
    ax1.set_xticks(range(len(etiquetas)))
    ax1.set_xticklabels(etiquetas, rotation=45)
    ax1.set_xlabel('Duración del Juego', size=10)
    ax1.set_ylabel('Promedio de Propietarios', color='skyblue', size=10)
    ax2.set_ylabel('Ratio de Valoraciones Positivas', color='red', size=10)
    
    # Añadir valores sobre las barras
    def format_value(value):
        if value >= 1e6:
            return f'{value/1e6:.1f}M'
        elif value >= 1e3:
            return f'{value/1e3:.1f}K'
        return f'{value:.0f}'
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            format_value(height),
            ha='center',
            va='bottom',
            color='darkblue'
        )
    
    # Añadir porcentajes en la línea de valoraciones
    for i, ratio in enumerate(stats['positive_ratio']):
        ax2.text(
            i,
            ratio,
            f'{ratio:.1%}',
            ha='center',
            va='bottom',
            color='red'
        )
    
    # Título y leyendas
    plt.title('Relación entre Tiempo de Juego, Propietarios y Valoraciones', 
              pad=20, size=14, weight='bold')
    
    # Añadir leyendas
    ax1.legend(bars, ['Propietarios'], loc='upper left')
    ax2.legend(line, ['Ratio valoraciones +'], loc='upper right')
    
    # Añadir grid suave
    ax1.grid(True, alpha=0.2)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    plt.savefig('graficos/1_dificultad_popularidad_simple.png', 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

# 2. Impacto de juegos Indie
def grafico_indie_vs_aaa(df):
    plt.figure(figsize=(12, 6))
    
    # Calcular estadísticas por grupo
    stats = df.groupby('is_indie').agg({
        'owners': ['mean', 'std', 'count'],
        'positive_ratio': 'mean'
    }).reset_index()
    
    # Calcular error estándar
    stats[('owners', 'se')] = stats[('owners', 'std')] / np.sqrt(stats[('owners', 'count')])
    
    # Crear gráfico de barras
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Barras para propietarios (eje y izquierdo)
    bars = ax1.bar(
        [0, 1],
        stats[('owners', 'mean')],
        yerr=stats[('owners', 'se')],
        capsize=5,
        color=['lightblue', 'darkblue'],
        alpha=0.7,
        width=0.5
    )
    
    # Configurar eje y derecho para ratio de valoraciones positivas
    ax2 = ax1.twinx()
    ax2.plot(
        [0, 1],
        stats[('positive_ratio', 'mean')],
        'ro-',
        linewidth=2,
        markersize=10,
        label='Ratio de valoraciones positivas'
    )
    
    # Configurar etiquetas y títulos
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['No Indie', 'Indie'])
    ax1.set_ylabel('Promedio de Propietarios')
    ax2.set_ylabel('Ratio de Valoraciones Positivas')
    
    plt.title('Comparación entre Juegos Indie y AAA\nPropietarios y Valoraciones', pad=20)
    
    # Añadir valores sobre las barras
    def format_value(value):
        if value >= 1e6:
            return f'{value/1e6:.1f}M'
        elif value >= 1e3:
            return f'{value/1e3:.1f}K'
        return f'{value:.0f}'
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            format_value(height),
            ha='center',
            va='bottom'
        )
    
    # Añadir leyendas
    ax1.legend(bars, ['Propietarios'], loc='upper left')
    ax2.legend(loc='upper right')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    plt.savefig('graficos/2_indie_vs_aaa.png', 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

# 3. Relación Precio-Éxito
def grafico_precio_exito(df):
    plt.figure(figsize=(12, 6))
    # Filtrar juegos de pago
    df_paid = df[df['price'] > 0]
    sns.regplot(data=df_paid, 
                x='price', 
                y='owners',
                scatter_kws={'alpha':0.5})
    plt.title('Relación entre Precio y Número de Propietarios')
    plt.xlabel('Precio ($)')
    plt.ylabel('Número de Propietarios')
    plt.savefig('graficos/3_precio_exito.png')
    plt.close()

# 4. Tendencias por Género
def grafico_tendencias_genero(df):
    # Separar géneros y contar
    generos = df['genres'].str.split(',', expand=True).stack()
    top_genres = generos.value_counts().head(10)
    
    plt.figure(figsize=(15, 6))
    sns.barplot(x=top_genres.index, y=top_genres.values)
    plt.title('Top 10 Géneros más Populares')
    plt.xticks(rotation=45)
    plt.xlabel('Género')
    plt.ylabel('Cantidad de Juegos')
    plt.savefig('graficos/4_tendencias_genero.png', bbox_inches='tight')
    plt.close()

# 5. Free-to-play vs Paid
def grafico_free_vs_paid(df):
    plt.figure(figsize=(12, 6))
    
    # Calcular estadísticas por grupo
    stats = df.groupby('is_free').agg({
        'owners': ['mean', 'std', 'count'],
        'positive_ratio': 'mean',
        'price': 'mean'
    }).reset_index()
    
    # Calcular error estándar
    stats[('owners', 'se')] = stats[('owners', 'std')] / np.sqrt(stats[('owners', 'count')])
    
    # Crear gráfico de barras
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Barras para propietarios (eje y izquierdo)
    bars = ax1.bar(
        [0, 1],
        stats[('owners', 'mean')],
        yerr=stats[('owners', 'se')],
        capsize=5,
        color=['lightblue', 'darkblue'],
        alpha=0.7,
        width=0.5
    )
    
    # Configurar eje y derecho para ratio de valoraciones positivas
    ax2 = ax1.twinx()
    ax2.plot(
        [0, 1],
        stats[('positive_ratio', 'mean')],
        'ro-',
        linewidth=2,
        markersize=10,
        label='Ratio de valoraciones positivas'
    )
    
    # Configurar etiquetas y títulos
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Free to Play', 'Paid'])
    ax1.set_ylabel('Promedio de Propietarios')
    ax2.set_ylabel('Ratio de Valoraciones Positivas')
    
    plt.title('Comparación entre Juegos Gratuitos y de Pago\nPropietarios y Valoraciones', 
              pad=20, size=14, weight='bold')
    
    # Añadir valores sobre las barras
    def format_value(value):
        if value >= 1e6:
            return f'{value/1e6:.1f}M'
        elif value >= 1e3:
            return f'{value/1e3:.1f}K'
        return f'{value:.0f}'
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            format_value(height),
            ha='center',
            va='bottom'
        )
    
    # Añadir precio promedio para juegos de pago
    avg_price = stats[stats['is_free'] == False][('price', 'mean')].values[0]
    ax1.text(1, 0, f'Precio promedio: ${avg_price:.2f}', 
             ha='center', va='bottom')
    
    # Añadir leyendas
    ax1.legend(bars, ['Propietarios'], loc='upper left')
    ax2.legend(loc='upper right')
    
    # Añadir grid
    ax1.grid(True, alpha=0.3)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    plt.savefig('graficos/5_free_vs_paid.png', 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

# 6. Influencia de Reseñas
def grafico_influencia_resenas(df):
    plt.figure(figsize=(10, 6))  # Reducido el tamaño de la figura
    
    # Filtrar datos extremos
    df_filtered = df[
        (df['positive_ratio'] > 0) & 
        (df['owners'] > 0) &
        (df['owners'] < df['owners'].quantile(0.95))  # Ajustado para mostrar menos outliers
    ]
    
    # Crear escala logarítmica para owners
    df_filtered['log_owners'] = np.log10(df_filtered['owners'])
    
    # Crear el scatter plot simplificado
    scatter = plt.scatter(
        df_filtered['positive_ratio'],
        df_filtered['log_owners'],
        c=df_filtered['price'],
        s=df_filtered['positive']/200,  # Reducido el tamaño de los puntos
        alpha=0.5,
        cmap='viridis'
    )
    
    # Añadir línea de tendencia
    z = np.polyfit(df_filtered['positive_ratio'], df_filtered['log_owners'], 1)
    p = np.poly1d(z)
    plt.plot(df_filtered['positive_ratio'], 
             p(df_filtered['positive_ratio']), 
             "r--", 
             alpha=0.8,
             label='Tendencia')
    
    # Configurar ejes
    plt.xlabel('Ratio de Valoraciones Positivas')
    plt.ylabel('Número de Propietarios (log)')
    
    # Ajustar etiquetas del eje Y
    def format_func(value, tick_number):
        return f'{10**value:,.0f}'
    
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    # Añadir colorbar compacta
    cbar = plt.colorbar(scatter, aspect=30)
    cbar.set_label('Precio ($)', rotation=270, labelpad=15)
    
    # Leyenda simplificada
    legend_elements = [
        plt.scatter([], [], s=50, c='gray', alpha=0.5, label='1K'),
        plt.scatter([], [], s=100, c='gray', alpha=0.5, label='5K'),
    ]
    
    plt.legend(handles=legend_elements, 
              title='Reseñas',
              loc='upper right')
    
    # Título simplificado
    plt.title('Relación entre Valoraciones y Éxito Comercial', 
             size=12, pad=10)
    
    # Grid suave
    plt.grid(True, alpha=0.2)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    plt.savefig('graficos/6_influencia_resenas.png', 
                bbox_inches='tight', 
                dpi=300,
                facecolor='white')
    plt.close()

# 7. Precio vs Metacritic
def grafico_precio_metacritic(df):
    plt.figure(figsize=(12, 6))
    # Filtrar juegos con puntuación de Metacritic
    df_metacritic = df[df['metacritic_score'] > 0]
    sns.regplot(data=df_metacritic, 
                x='price', 
                y='metacritic_score',
                scatter_kws={'alpha':0.5})
    plt.title('Relación entre Precio y Puntuación de Metacritic')
    plt.xlabel('Precio ($)')
    plt.ylabel('Puntuación Metacritic')
    plt.savefig('graficos/7_precio_metacritic.png')
    plt.close()

# 8. Early Access
def grafico_early_access(df):
    plt.figure(figsize=(12, 6))
    
    # Asegurarnos de que has_early_access sea booleano
    df['has_early_access'] = df['has_early_access'].fillna(False)
    
    # Calcular estadísticas por grupo
    stats = df.groupby('has_early_access').agg({
        'owners': ['mean', 'std', 'count'],
        'positive_ratio': 'mean'
    }).reset_index()
    
    # Verificar que tenemos datos para ambos grupos
    if len(stats) < 2:
        print("No hay suficientes datos para comparar Early Access vs No Early Access")
        return
    
    # Calcular error estándar
    stats[('owners', 'se')] = stats[('owners', 'std')] / np.sqrt(stats[('owners', 'count')])
    
    # Crear gráfico con dos ejes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Barras para propietarios (eje y izquierdo)
    bars = ax1.bar(
        [0, 1],
        stats[('owners', 'mean')],
        yerr=stats[('owners', 'se')],
        capsize=5,
        color=['lightblue', 'darkblue'],
        alpha=0.7,
        width=0.5
    )
    
    # Línea para ratio de valoraciones (eje y derecho)
    ax2 = ax1.twinx()
    line = ax2.plot(
        [0, 1],
        stats['positive_ratio', 'mean'].values,  # Asegurarnos de obtener los valores
        'ro-',
        linewidth=2,
        markersize=8,
        label='Ratio de valoraciones positivas'
    )
    
    # Configurar ejes y etiquetas
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['No Early Access', 'Early Access'])
    ax1.set_ylabel('Promedio de Propietarios', color='darkblue')
    ax2.set_ylabel('Ratio de Valoraciones Positivas', color='red')
    
    # Añadir valores sobre las barras
    def format_value(value):
        if value >= 1e6:
            return f'{value/1e6:.1f}M'
        elif value >= 1e3:
            return f'{value/1e3:.1f}K'
        return f'{value:.0f}'
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            format_value(height),
            ha='center',
            va='bottom'
        )
    
    # Añadir porcentajes en la línea
    for i, ratio in enumerate(stats[('positive_ratio', 'mean')]):
        ax2.text(
            i,
            ratio,
            f'{ratio:.1%}',
            ha='center',
            va='bottom',
            color='red'
        )
    
    # Título y leyendas
    plt.title('Comparación: Early Access vs Lanzamiento Tradicional\nPropietarios y Valoraciones', 
              pad=20, size=14, weight='bold')
    
    # Añadir leyendas
    ax1.legend(bars, ['Propietarios'], loc='upper left')
    ax2.legend(line, ['Ratio valoraciones +'], loc='upper right')
    
    # Añadir grid
    ax1.grid(True, alpha=0.3)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    plt.savefig('graficos/8_early_access.png', 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def main():
    crear_directorio()
    
    # Preprocesar datos
    df_processed = preprocesar_datos(df)
    
    # Generar gráficos
    grafico_dificultad_popularidad_simple(df_processed)
    grafico_indie_vs_aaa(df_processed)
    grafico_precio_exito(df_processed)
    grafico_tendencias_genero(df_processed)
    grafico_free_vs_paid(df_processed)
    grafico_influencia_resenas(df_processed)
    grafico_precio_metacritic(df_processed)
    grafico_early_access(df_processed)
    
    print("Todos los gráficos han sido generados en el directorio 'graficos'")

if __name__ == "__main__":
    main()
