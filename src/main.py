import pandas as pd
import numpy as np

# ============================================
# CONFIGURACIÓN
# ============================================
EXPORTAR_LIMPIO = True  # Cambia a False si NO quieres exportar el CSV
TAMANO_MUESTRA = 100000  # Cantidad de canciones para la muestra final

# ============================================
# CARGAR DATOS
# ============================================
print("📀 Cargando dataset...")
df = pd.read_csv('./../data/rawuniversal_top_spotify_songs.csv')
print(f"✅ Dataset cargado: {len(df):,} canciones")

# ============================================
# LAS 11 FEATURES QUE USAREMOS (según tu tabla)
# ============================================
features = [
    'danceability',    # float (0-1) - Qué tan bailable es
    'energy',          # float (0-1) - Intensidad/actividad
    'valence',         # float (0-1) - Positividad/alegría
    'acousticness',    # float (0-1) - Acústica vs electrónica
    'instrumentalness', # float (0-1) - Si no tiene voz
    'liveness',        # float (0-1) - Si tiene audiencia en vivo
    'speechiness',     # float (0-1) - Cantidad de palabras habladas
    'tempo',           # float (BPM) - Velocidad de la canción
    'loudness',        # float (dB) - Volumen (negativo a positivo)
    'duration_ms',     # int (ms) - Duración
    'mode'             # int (0 o 1) - Mayor o menor
]

# Verificar que todas las columnas existen en el dataset
columnas_faltantes = [col for col in features if col not in df.columns]
if columnas_faltantes:
    print(f"⚠️ Atención: Estas columnas NO existen en el dataset: {columnas_faltantes}")
else:
    print(f"✅ Las 11 features existen en el dataset")

# ============================================
# 1. DIAGNÓSTICO COMPLETO
# ============================================
print("\n" + "="*60)
print("🔍 DIAGNÓSTICO COMPLETO DEL DATASET")
print("="*60)

# 1.1 Valores nulos
print("\n📌 1. VALORES NULOS:")
print("-"*40)
nulos = df[features].isnull().sum()
if nulos.sum() == 0:
    print("✅ No hay valores nulos en ninguna feature")
else:
    print(nulos[nulos > 0])

# 1.2 Valores imposibles
print("\n📌 2. VALORES IMPOSIBLES:")
print("-"*40)
tempo_cero = df[df['tempo'] == 0]
print(f"⚠️ tempo = 0: {len(tempo_cero)} canciones")

# Verificar mode (debe ser 0 o 1)
mode_fuera = df[(df['mode'] != 0) & (df['mode'] != 1)]
if len(mode_fuera) > 0:
    print(f"⚠️ mode fuera de rango (0 o 1): {len(mode_fuera)} canciones")

# Verificar duration_ms > 0
duration_cero = df[df['duration_ms'] <= 0]
if len(duration_cero) > 0:
    print(f"⚠️ duration_ms <= 0: {len(duration_cero)} canciones")

# 1.3 Duplicados exactos
print("\n📌 3. DUPLICADOS EXACTOS (misma fila en todas las columnas):")
print("-"*40)
duplicados = df.duplicated().sum()
print(f"⚠️ Filas duplicadas: {duplicados}")

# 1.4 Tipos de datos
print("\n📌 4. TIPOS DE DATOS:")
print("-"*40)
for col in features:
    print(f"   {col}: {df[col].dtype}")

# ============================================
# 2. LIMPIEZA
# ============================================
print("\n" + "="*60)
print("🧹 APLICANDO LIMPIEZA")
print("="*60)

df_limpio = df.copy()

# Eliminar tempo = 0
antes = len(df_limpio)
df_limpio = df_limpio[df_limpio['tempo'] > 0]
print(f"✅ Eliminadas {antes - len(df_limpio)} canciones con tempo = 0")

# Eliminar mode que no sea 0 o 1 (si existe)
antes = len(df_limpio)
df_limpio = df_limpio[(df_limpio['mode'] == 0) | (df_limpio['mode'] == 1)]
print(f"✅ Eliminadas {antes - len(df_limpio)} canciones con mode inválido")

# Eliminar duration_ms <= 0
antes = len(df_limpio)
df_limpio = df_limpio[df_limpio['duration_ms'] > 0]
print(f"✅ Eliminadas {antes - len(df_limpio)} canciones con duration_ms <= 0")

# Eliminar duplicados exactos
antes = len(df_limpio)
df_limpio = df_limpio.drop_duplicates()
print(f"✅ Eliminadas {antes - len(df_limpio)} filas duplicadas")

# Eliminar nulos en las features
antes = len(df_limpio)
df_limpio = df_limpio.dropna(subset=features)
print(f"✅ Eliminadas {antes - len(df_limpio)} canciones con valores nulos")

print(f"\n📊 RESULTADO FINAL: {len(df_limpio):,} canciones listas para clustering")

# ============================================
# 3. EXPORTAR CSV LIMPIO (OPCIONAL)
# ============================================
if EXPORTAR_LIMPIO:
    print("\n" + "="*60)
    print("💾 EXPORTANDO DATASET LIMPIO")
    print("="*60)
    
    # Exportar completo
    df_limpio.to_csv('./../data/processed/universal_top_spotify_songs_clean.csv', index=False)
    print("✅ Exportado: universal_top_spotify_songs_clean.csv")
    
    # Exportar solo las 11 features
    df_features = df_limpio[features]
    df_features.to_csv('./../data/processed/spotify_features_only.csv', index=False)
    print("✅ Exportado: spotify_features_only.csv (solo las 11 features)")
else:
    print("\n⏭️ Exportación desactivada (EXPORTAR_LIMPIO = False)")

# ============================================
# 4. MUESTRA PARA CLUSTERING
# ============================================
print("\n" + "="*60)
print("🎯 MUESTRA PARA CLUSTERING")
print("="*60)

# Tomar muestra aleatoria
df_muestra = df_limpio.sample(n=min(TAMANO_MUESTRA, len(df_limpio)), random_state=42)

# Escalar los datos (necesario para clustering)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_muestra[features])

print(f"✅ Muestra tomada: {len(df_muestra):,} canciones")
print(f"✅ Features escaladas: {X_scaled.shape[1]} variables")
print(f"✅ Escalador guardado (usar mismo para nuevos datos)")

# Exportar muestra
if EXPORTAR_LIMPIO:
    # Muestra original (sin escalar)
    df_muestra.to_csv('./../data/processed/spotify_muestra_100k.csv', index=False)
    print("✅ Exportado: spotify_muestra_100k.csv (datos sin escalar)")
    
    # Muestra escalada
    df_muestra_scaled = pd.DataFrame(X_scaled, columns=features)
    df_muestra_scaled.to_csv('./../data/processed/spotify_muestra_100k_scaled.csv', index=False)
    print("✅ Exportado: spotify_muestra_100k_scaled.csv (datos escalados, listos para clustering)")

# ============================================
# 5. RESUMEN FINAL
# ============================================
print("\n" + "="*60)
print("📋 RESUMEN FINAL")
print("="*60)
print(f"""
─────────────────────────────────────────────────────────────
\tDATASET ORIGINAL:\t{2_110_316:>12,} canciones
\tDATASET LIMPIO:\t{len(df_limpio):>12,} canciones
\tMUESTRA PARA ML:\t{len(df_muestra):>12,} canciones
\tFEATURES USADAS:\t{len(features):>12} variables
─────────────────────────────────────────────────────────────
\tLISTA DE FEATURES:
\t1. danceability     7. liveness
\t2. energy           8. speechiness
\t3. valence          9. tempo
\t4. acousticness     10. loudness
\t5. instrumentalness 11. duration_ms
\t6. mode
─────────────────────────────────────────────────────────────
\t✅ Dataset listo para aplicar los 3 algoritmos:
\t\t- K-Means
\t\t- DBSCAN
\t\t- Agglomerative Clustering
─────────────────────────────────────────────────────────────
""")

# ============================================
# 6. ESTADÍSTICAS DE LA MUESTRA
# ============================================
print("\n📊 ESTADÍSTICAS DE LA MUESTRA (datos sin escalar):")
print("-"*40)
print(df_muestra[features].describe().round(3))

# ============================================
# 7. VERIFICACIÓN FINAL DE RANGOS
# ============================================
print("\n📌 VERIFICACIÓN DE RANGOS ESPERADOS:")
print("-"*40)

rangos = {
    'danceability': (0, 1),
    'energy': (0, 1),
    'valence': (0, 1),
    'acousticness': (0, 1),
    'instrumentalness': (0, 1),
    'liveness': (0, 1),
    'speechiness': (0, 1),
    'tempo': (0, 300),
    'loudness': (-60, 10),
    'duration_ms': (1000, 600000),  # 1 segundo a 10 minutos
    'mode': (0, 1)
}

for feature, (min_esp, max_esp) in rangos.items():
    valores = df_muestra[feature]
    fuera = valores[(valores < min_esp) | (valores > max_esp)]
    if len(fuera) > 0:
        print(f"⚠️ {feature}: {len(fuera)} valores fuera de rango [{min_esp}, {max_esp}]")
    else:
        print(f"✅ {feature}: todos en rango [{min_esp}, {max_esp}]")