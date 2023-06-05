 !pip install kneed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from kneed import KneeLocator

# Leer los datos desde un archivo Excel
data = pd.read_excel('', names=['x', 'y', 'z'])

# Convertir los datos en una matriz de numpy
X = np.array(data)

# Definir el rango de clusters a probar
k_range = range(2, 21)

# Ejecutar el algoritmo K-means para cada valor de k y almacenar los resultados
sse = []
labels_dict = {}
centroids_dict = {}
for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
    labels_dict[k] = kmeans.labels_
    centroids_dict[k] = kmeans.cluster_centers_
    
# Obtener el valor óptimo de k según el método del codo
kl = KneeLocator(k_range, sse, curve='convex', direction='decreasing')
best_k = kl.elbow
# Obtener los puntos de cada cluster y los centroides correspondientes
labels = labels_dict[best_k]
centroids = centroids_dict[best_k]

# Imprimir los puntos de cada cluster y los centroides correspondientes
for i in range(best_k):
    cluster_points = [X[j] for j in range(len(X)) if labels[j] == i]
    print(f'Cluster {i+1}:')
    for j, point in enumerate(cluster_points):
        index = data[(data['x']==point[0]) & (data['y']==point[1]) & (data['z']==point[2])].index[0] + 1
        print(f'Transition {index-1}: {point}')
    print(f'Centroid: {centroids[i]}')
    print()

# Gráfica del método del codo
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_range, sse, 'bx-')
ax.set_xlabel('Cluster number')
ax.set_ylabel('SSE')
ax.set_title('Elbow method')
ax.set_xticks(range(2, 21))
ax.vlines(x=best_k, ymin=min(sse), ymax=max(sse), linestyles='dashed')
plt.show()

# Gráfica 3D de los puntos y los centroides de cada cluster
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(best_k):
    cluster_points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
    ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2])
    ax.scatter(centroids[i,0], centroids[i,1], centroids[i,2], s=100, marker='x', color='grey', linewidth=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Clustering K-means')
plt.show()