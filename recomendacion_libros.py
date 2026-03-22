"""
Hecho por : Rocha Cantu Nidia Wendoly  Fecha: 22  de Marzo 2026
Clase: Inteligencia artificial y su ética - Tema 4.5 Minería de Datos - Actividad 25
MIA - Intituto Tecnológico de Nuevo Laredo - Prof. Carlos Arturo Guerrero Crespo
Titulo: Sistema de Recomendación Basado en Reglas de Asociación
Descripción:
Analiza patrones de compra y recomienda productos relacionados automáticamente.
"""

# Librerias
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Simular transacciones
transacciones = [
    ['Harry Potter', 'El Principito'],
    ['1984', 'Fahrenheit 451'],
    ['Harry Potter', 'Crónicas de Narnia'],
    ['La Sombra del Viento', 'El Nombre de la Rosa'],
    ['1984', 'Un Mundo Feliz']
]

# One-Hot Encoding
te = TransactionEncoder()
te_ary = te.fit(transacciones).transform(transacciones)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori
frecuentes = apriori(df, min_support=0.05, use_colnames=True)
reglas = association_rules(frecuentes, metric="lift", min_threshold=1.2)

# Imprimir reglas
for _, row in reglas.iterrows():
    print(f"{list(row['antecedents'])} -> {list(row['consequents'])} | support: {row['support']:.2f}, lift: {row['lift']:.2f}")

# Función de recomendación
def recomendar(carrito):
    recomendaciones = []
    for _, row in reglas.iterrows():
        if all(item in carrito for item in row['antecedents']):
            recomendaciones.extend(list(row['consequents']))
    return list(set(recomendaciones))

# Ejemplo
print(recomendar(['Harry Potter']))