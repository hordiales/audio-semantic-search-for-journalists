"""
Visualizador de mapas de calor semánticos para análisis de embeddings de audio.
Permite visualizar similitudes semánticas entre segmentos de audio y consultas.
"""

import logging
from pathlib import Path
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

warnings.filterwarnings('ignore')

# Imports para clustering y análisis
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Imports condicionales para visualizaciones avanzadas
try:
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Imports locales
try:
    from .embedding_evaluation_framework import EmbeddingBenchmark
    from .test_data_generator import SyntheticTestDataGenerator
except ImportError:
    try:
        from embedding_evaluation_framework import EmbeddingBenchmark
        from test_data_generator import SyntheticTestDataGenerator
    except ImportError:
        pass

logger = logging.getLogger(__name__)


class SemanticHeatmapVisualizer:
    """
    Visualizador de mapas de calor semánticos para embeddings de audio
    """

    def __init__(self, output_dir: str = "semantic_heatmap_results"):
        """
        Inicializa el visualizador de mapas de calor semánticos

        Args:
            output_dir: Directorio de salida para visualizaciones
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Configurar estilo de matplotlib
        plt.style.use('default')
        sns.set_palette("viridis")

        logger.info("🔥 Visualizador de mapas de calor semánticos inicializado")
        logger.info(f"📁 Salida en: {self.output_dir}")

    def calculate_similarity_matrix(self, embeddings: np.ndarray,
                                  metric: str = "cosine") -> np.ndarray:
        """
        Calcula matriz de similitud entre embeddings

        Args:
            embeddings: Array de embeddings (n_samples, n_features)
            metric: Métrica de similitud ("cosine", "euclidean", "manhattan")

        Returns:
            Matriz de similitud (n_samples, n_samples)
        """
        if metric == "cosine":
            # Normalizar embeddings para similitud coseno
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        elif metric == "euclidean":
            # Convertir distancias a similitudes
            distances = euclidean_distances(embeddings)
            similarity_matrix = 1 / (1 + distances)
        elif metric == "manhattan":
            from sklearn.metrics.pairwise import manhattan_distances
            distances = manhattan_distances(embeddings)
            similarity_matrix = 1 / (1 + distances)
        else:
            raise ValueError(f"Métrica no soportada: {metric}")

        return similarity_matrix

    def create_basic_heatmap(self, similarity_matrix: np.ndarray,
                           labels: list[str] | None = None,
                           title: str = "Mapa de Calor Semántico",
                           save_path: str | None = None) -> plt.Figure:
        """
        Crea mapa de calor básico con matplotlib/seaborn

        Args:
            similarity_matrix: Matriz de similitud
            labels: Etiquetas para ejes
            title: Título del gráfico
            save_path: Ruta para guardar el gráfico

        Returns:
            Figura de matplotlib
        """
        # Configurar tamaño según dimensión de la matriz
        size = max(8, min(20, len(similarity_matrix) * 0.5))
        fig, ax = plt.subplots(figsize=(size, size))

        # Crear heatmap
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)

        heatmap = sns.heatmap(
            similarity_matrix,
            mask=mask,
            annot=len(similarity_matrix) <= 20,  # Solo anotar si no es muy grande
            fmt='.2f',
            cmap='viridis',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .8},
            ax=ax
        )

        # Configurar etiquetas
        if labels:
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels, rotation=0)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Segmentos de Audio', fontweight='bold')
        ax.set_ylabel('Segmentos de Audio', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"🔥 Mapa de calor guardado en {save_path}")

        return fig

    def create_interactive_heatmap(self, similarity_matrix: np.ndarray,
                                 labels: list[str] | None = None,
                                 metadata: list[dict] | None = None,
                                 title: str = "Mapa de Calor Semántico Interactivo",
                                 save_path: str | None = None):
        """
        Crea mapa de calor interactivo con Plotly

        Args:
            similarity_matrix: Matriz de similitud
            labels: Etiquetas para ejes
            metadata: Metadatos adicionales para hover
            title: Título del gráfico
            save_path: Ruta para guardar como HTML
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("⚠️  Plotly no disponible, creando mapa de calor básico")
            return self.create_basic_heatmap(similarity_matrix, labels, title, save_path)

        # Preparar datos de hover
        hover_text = []
        for i in range(len(similarity_matrix)):
            hover_row = []
            for j in range(len(similarity_matrix)):
                base_text = f"Similitud: {similarity_matrix[i,j]:.3f}"

                if labels:
                    base_text += f"<br>X: {labels[j]}<br>Y: {labels[i]}"

                if metadata and i < len(metadata) and j < len(metadata):
                    base_text += f"<br>Categoría X: {metadata[j].get('category', 'N/A')}"
                    base_text += f"<br>Categoría Y: {metadata[i].get('category', 'N/A')}"

                hover_row.append(base_text)
            hover_text.append(hover_row)

        # Crear figura interactiva
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels if labels else list(range(len(similarity_matrix))),
            y=labels if labels else list(range(len(similarity_matrix))),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text,
            colorscale='Viridis',
            colorbar={
                'title': "Similitud Semántica",
                'titleside': "right"
            }
        ))

        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title="Segmentos de Audio",
            yaxis_title="Segmentos de Audio",
            width=800,
            height=800
        )

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_html(save_path + '.html')
            logger.info(f"🔥 Mapa de calor interactivo guardado en {save_path}")

        return fig

    def create_clustered_heatmap(self, similarity_matrix: np.ndarray,
                                labels: list[str] | None = None,
                                clustering_method: str = "hierarchical",
                                n_clusters: int = 5,
                                title: str = "Mapa de Calor con Clustering",
                                save_path: str | None = None) -> tuple[plt.Figure, np.ndarray]:
        """
        Crea mapa de calor con clustering jerárquico

        Args:
            similarity_matrix: Matriz de similitud
            labels: Etiquetas originales
            clustering_method: Método de clustering ("hierarchical", "kmeans", "dbscan")
            n_clusters: Número de clusters (para métodos que lo requieren)
            title: Título del gráfico
            save_path: Ruta para guardar

        Returns:
            Tupla con (figura, índices de ordenamiento)
        """
        # Convertir similitud a distancia
        distance_matrix = 1 - similarity_matrix

        # Asegurar que la diagonal sea exactamente cero
        np.fill_diagonal(distance_matrix, 0.0)

        if clustering_method == "hierarchical":
            # Clustering jerárquico
            linkage_matrix = linkage(squareform(distance_matrix), method='ward')

            # Crear figura con dendrogram y heatmap
            fig = plt.figure(figsize=(15, 12))

            # Dendrogram superior
            ax_dendro = plt.subplot2grid((4, 4), (0, 1), colspan=3)
            dendro = dendrogram(linkage_matrix, ax=ax_dendro, orientation='top')
            ax_dendro.set_xticks([])
            ax_dendro.set_yticks([])

            # Obtener orden de clustering
            cluster_order = dendro['leaves']

            # Dendrogram izquierdo
            ax_dendro_left = plt.subplot2grid((4, 4), (1, 0), rowspan=3)
            dendrogram(linkage_matrix, ax=ax_dendro_left, orientation='left')
            ax_dendro_left.set_xticks([])
            ax_dendro_left.set_yticks([])

        elif clustering_method == "kmeans":
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(1 - similarity_matrix)
            cluster_order = np.argsort(cluster_labels)

            fig = plt.figure(figsize=(12, 10))

        elif clustering_method == "dbscan":
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
            cluster_labels = dbscan.fit_predict(distance_matrix)
            cluster_order = np.argsort(cluster_labels)

            fig = plt.figure(figsize=(12, 10))

        else:
            raise ValueError(f"Método de clustering no soportado: {clustering_method}")

        # Reordenar matriz según clustering
        clustered_matrix = similarity_matrix[np.ix_(cluster_order, cluster_order)]

        # Crear heatmap principal
        if clustering_method == "hierarchical":
            ax_heatmap = plt.subplot2grid((4, 4), (1, 1), rowspan=3, colspan=3)
        else:
            ax_heatmap = plt.subplot(111)

        # Etiquetas reordenadas
        clustered_labels = None
        if labels:
            clustered_labels = [labels[i] for i in cluster_order]

        heatmap = sns.heatmap(
            clustered_matrix,
            annot=len(clustered_matrix) <= 15,
            fmt='.2f',
            cmap='viridis',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .8},
            ax=ax_heatmap
        )

        if clustered_labels:
            ax_heatmap.set_xticklabels(clustered_labels, rotation=45, ha='right')
            ax_heatmap.set_yticklabels(clustered_labels, rotation=0)

        ax_heatmap.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"🔥 Mapa de calor con clustering guardado en {save_path}")

        return fig, cluster_order

    def create_semantic_landscape(self, embeddings: np.ndarray,
                                labels: list[str] | None = None,
                                metadata: list[dict] | None = None,
                                method: str = "tsne",
                                title: str = "Paisaje Semántico 2D",
                                save_path: str | None = None) -> plt.Figure:
        """
        Crea visualización 2D del paisaje semántico usando reducción de dimensionalidad

        Args:
            embeddings: Embeddings originales
            labels: Etiquetas para puntos
            metadata: Metadatos para coloring
            method: Método de reducción ("tsne", "pca", "umap")
            title: Título del gráfico
            save_path: Ruta para guardar

        Returns:
            Figura de matplotlib
        """
        # Reducción de dimensionalidad
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                logger.warning("⚠️  UMAP no disponible, usando t-SNE")
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        else:
            raise ValueError(f"Método no soportado: {method}")

        # Aplicar reducción
        embeddings_2d = reducer.fit_transform(embeddings)

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 10))

        # Determinar colores por categoría si hay metadata
        colors = None
        if metadata:
            categories = [item.get('category', 'unknown') for item in metadata]
            unique_categories = list(set(categories))
            color_map = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
            colors = [color_map[unique_categories.index(cat)] for cat in categories]

        # Scatter plot
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=colors if colors else 'viridis',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

        # Añadir etiquetas si no son demasiadas
        if labels and len(labels) <= 50:
            for i, label in enumerate(labels):
                ax.annotate(
                    label[:20] + "..." if len(label) > 20 else label,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )

        # Configurar gráfico
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} Dimensión 1', fontweight='bold')
        ax.set_ylabel(f'{method.upper()} Dimensión 2', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Leyenda por categorías
        if metadata and colors:
            unique_categories = list({item.get('category', 'unknown') for item in metadata})
            legend_elements = []
            for i, cat in enumerate(unique_categories):
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=color_map[i], markersize=10, label=cat)
                )
            ax.legend(handles=legend_elements, title='Categorías', loc='best')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"🗺️  Paisaje semántico guardado en {save_path}")

        return fig

    def create_query_similarity_heatmap(self, query_embeddings: np.ndarray,
                                      document_embeddings: np.ndarray,
                                      query_labels: list[str],
                                      document_labels: list[str],
                                      title: str = "Similitud Consulta-Documento",
                                      save_path: str | None = None) -> plt.Figure:
        """
        Crea mapa de calor de similitud entre consultas y documentos

        Args:
            query_embeddings: Embeddings de consultas
            document_embeddings: Embeddings de documentos
            query_labels: Etiquetas de consultas
            document_labels: Etiquetas de documentos
            title: Título del gráfico
            save_path: Ruta para guardar

        Returns:
            Figura de matplotlib
        """
        # Calcular matriz de similitud cruzada
        similarity_matrix = cosine_similarity(query_embeddings, document_embeddings)

        # Crear figura
        fig, ax = plt.subplots(figsize=(max(8, len(document_labels) * 0.3),
                                       max(6, len(query_labels) * 0.5)))

        # Crear heatmap
        heatmap = sns.heatmap(
            similarity_matrix,
            xticklabels=document_labels,
            yticklabels=query_labels,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0.5,
            linewidths=0.5,
            cbar_kws={"shrink": .8, "label": "Similitud Coseno"},
            ax=ax
        )

        # Configurar etiquetas
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Documentos de Audio', fontweight='bold')
        ax.set_ylabel('Consultas', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"🔍 Mapa de similitud consulta-documento guardado en {save_path}")

        return fig

    def create_comprehensive_semantic_analysis(self, embeddings: np.ndarray,
                                             labels: list[str],
                                             metadata: list[dict],
                                             queries: list[str] | None = None,
                                             query_embeddings: np.ndarray = None,
                                             output_prefix: str = "semantic_analysis") -> dict[str, Any]:
        """
        Crea análisis semántico comprehensivo con múltiples visualizaciones

        Args:
            embeddings: Embeddings de documentos
            labels: Etiquetas de documentos
            metadata: Metadatos de documentos
            queries: Consultas de texto (opcional)
            query_embeddings: Embeddings de consultas (opcional)
            output_prefix: Prefijo para archivos de salida

        Returns:
            Diccionario con rutas de archivos generados y métricas
        """
        results = {
            "files_generated": [],
            "metrics": {},
            "clusters": {}
        }

        logger.info("🔥 Iniciando análisis semántico comprehensivo...")

        # 1. Mapa de calor básico de similitud
        similarity_matrix = self.calculate_similarity_matrix(embeddings)

        basic_heatmap_path = self.output_dir / f"{output_prefix}_basic_heatmap.png"
        self.create_basic_heatmap(
            similarity_matrix,
            labels,
            "Mapa de Calor de Similitud Semántica",
            str(basic_heatmap_path)
        )
        results["files_generated"].append(str(basic_heatmap_path))

        # 2. Mapa de calor interactivo (si Plotly disponible)
        if PLOTLY_AVAILABLE:
            interactive_heatmap_path = self.output_dir / f"{output_prefix}_interactive_heatmap.html"
            self.create_interactive_heatmap(
                similarity_matrix,
                labels,
                metadata,
                "Mapa de Calor Semántico Interactivo",
                str(interactive_heatmap_path)
            )
            results["files_generated"].append(str(interactive_heatmap_path))

        # 3. Mapa de calor con clustering
        clustered_heatmap_path = self.output_dir / f"{output_prefix}_clustered_heatmap.png"
        _fig_clustered, cluster_order = self.create_clustered_heatmap(
            similarity_matrix,
            labels,
            "hierarchical",
            title="Mapa de Calor con Clustering Jerárquico",
            save_path=str(clustered_heatmap_path)
        )
        results["files_generated"].append(str(clustered_heatmap_path))
        if hasattr(cluster_order, 'tolist'):
            results["clusters"]["hierarchical_order"] = cluster_order.tolist()
        else:
            results["clusters"]["hierarchical_order"] = list(cluster_order)

        # 4. Paisaje semántico 2D
        landscape_path = self.output_dir / f"{output_prefix}_semantic_landscape.png"
        self.create_semantic_landscape(
            embeddings,
            labels,
            metadata,
            "tsne",
            "Paisaje Semántico (t-SNE)",
            str(landscape_path)
        )
        results["files_generated"].append(str(landscape_path))

        # 5. Análisis de consultas (si se proporcionan)
        if queries and query_embeddings is not None:
            query_similarity_path = self.output_dir / f"{output_prefix}_query_similarity.png"
            self.create_query_similarity_heatmap(
                query_embeddings,
                embeddings,
                queries,
                labels,
                "Similitud Consulta-Documento",
                str(query_similarity_path)
            )
            results["files_generated"].append(str(query_similarity_path))

        # 6. Métricas de análisis
        results["metrics"] = {
            "mean_similarity": float(np.mean(similarity_matrix)),
            "std_similarity": float(np.std(similarity_matrix)),
            "max_similarity": float(np.max(similarity_matrix)),
            "min_similarity": float(np.min(similarity_matrix)),
            "num_documents": len(embeddings),
            "embedding_dimension": embeddings.shape[1]
        }

        # 7. Análisis por categorías
        if metadata:
            category_analysis = self._analyze_by_categories(similarity_matrix, metadata)
            results["metrics"]["category_analysis"] = category_analysis

        logger.info(f"✅ Análisis semántico completado. {len(results['files_generated'])} archivos generados.")

        return results

    def _analyze_by_categories(self, similarity_matrix: np.ndarray,
                              metadata: list[dict]) -> dict[str, Any]:
        """
        Analiza similitudes por categorías

        Args:
            similarity_matrix: Matriz de similitud
            metadata: Metadatos con categorías

        Returns:
            Análisis por categorías
        """
        categories = [item.get('category', 'unknown') for item in metadata]
        unique_categories = list(set(categories))

        analysis = {
            "intra_category_similarity": {},
            "inter_category_similarity": {},
            "category_counts": {}
        }

        for cat in unique_categories:
            # Índices de documentos de esta categoría
            cat_indices = [i for i, c in enumerate(categories) if c == cat]
            analysis["category_counts"][cat] = len(cat_indices)

            if len(cat_indices) > 1:
                # Similitud intra-categoría
                intra_similarities = []
                for i in range(len(cat_indices)):
                    for j in range(i+1, len(cat_indices)):
                        idx_i, idx_j = cat_indices[i], cat_indices[j]
                        intra_similarities.append(similarity_matrix[idx_i, idx_j])

                analysis["intra_category_similarity"][cat] = {
                    "mean": float(np.mean(intra_similarities)),
                    "std": float(np.std(intra_similarities)),
                    "count": len(intra_similarities)
                }

        # Similitud inter-categoría
        for i, cat1 in enumerate(unique_categories):
            for j, cat2 in enumerate(unique_categories[i+1:], i+1):
                cat1_indices = [idx for idx, c in enumerate(categories) if c == cat1]
                cat2_indices = [idx for idx, c in enumerate(categories) if c == cat2]

                inter_similarities = []
                for idx1 in cat1_indices:
                    for idx2 in cat2_indices:
                        inter_similarities.append(similarity_matrix[idx1, idx2])

                key = f"{cat1}_vs_{cat2}"
                analysis["inter_category_similarity"][key] = {
                    "mean": float(np.mean(inter_similarities)),
                    "std": float(np.std(inter_similarities)),
                    "count": len(inter_similarities)
                }

        return analysis

    def create_category_comparison_heatmap(self, similarity_matrix: np.ndarray,
                                         metadata: list[dict],
                                         title: str = "Similitud por Categorías",
                                         save_path: str | None = None) -> plt.Figure:
        """
        Crea mapa de calor agregado por categorías

        Args:
            similarity_matrix: Matriz de similitud
            metadata: Metadatos con categorías
            title: Título del gráfico
            save_path: Ruta para guardar

        Returns:
            Figura de matplotlib
        """
        categories = [item.get('category', 'unknown') for item in metadata]
        unique_categories = sorted(set(categories))

        # Crear matriz agregada por categorías
        category_matrix = np.zeros((len(unique_categories), len(unique_categories)))

        for i, cat1 in enumerate(unique_categories):
            for j, cat2 in enumerate(unique_categories):
                cat1_indices = [idx for idx, c in enumerate(categories) if c == cat1]
                cat2_indices = [idx for idx, c in enumerate(categories) if c == cat2]

                similarities = []
                for idx1 in cat1_indices:
                    for idx2 in cat2_indices:
                        similarities.append(similarity_matrix[idx1, idx2])

                category_matrix[i, j] = np.mean(similarities) if similarities else 0

        # Crear visualización
        fig, ax = plt.subplots(figsize=(10, 8))

        heatmap = sns.heatmap(
            category_matrix,
            xticklabels=unique_categories,
            yticklabels=unique_categories,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .8, "label": "Similitud Promedio"},
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Categorías', fontweight='bold')
        ax.set_ylabel('Categorías', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Mapa de categorías guardado en {save_path}")

        return fig


def create_demo_semantic_heatmap():
    """Función demo para mostrar capacidades del visualizador"""
    print("🔥 Demo: Mapa de Calor Semántico")
    print("=" * 50)

    try:
        # Importar generador de datos
        import sys
        sys.path.insert(0, 'src')
        from test_data_generator import SyntheticTestDataGenerator

        # Generar datos sintéticos
        generator = SyntheticTestDataGenerator("demo_heatmap_output")
        df = generator.generate_test_dataset(num_samples=20)

        # Simular embeddings (en aplicación real vendrían de modelos)
        np.random.seed(42)
        n_samples = len(df)
        embedding_dim = 128

        # Crear embeddings con estructura semántica
        embeddings = []
        categories = df['category'].unique()
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}

        for _, row in df.iterrows():
            # Base embedding con componente categórica
            cat_idx = cat_to_idx[row['category']]
            base_embedding = np.random.normal(0, 1, embedding_dim)

            # Añadir componente categórica
            base_embedding[cat_idx * 20:(cat_idx + 1) * 20] += 2.0

            # Normalizar
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            embeddings.append(base_embedding)

        embeddings = np.array(embeddings)

        # Crear visualizador
        visualizer = SemanticHeatmapVisualizer("demo_heatmap_output")

        # Preparar metadatos
        metadata = []
        labels = []
        for _, row in df.iterrows():
            metadata.append({
                'category': row['category'],
                'speaker': row['speaker_id'],
                'duration': row['duration']
            })
            labels.append(f"{row['category'][:3]}_{row['id']}")

        # Crear análisis comprehensivo
        results = visualizer.create_comprehensive_semantic_analysis(
            embeddings,
            labels,
            metadata,
            output_prefix="demo"
        )

        # Mostrar resultados
        print("✅ Análisis completado")
        print(f"📁 Archivos generados: {len(results['files_generated'])}")
        for file_path in results['files_generated']:
            print(f"   📄 {file_path}")

        print("\n📊 Métricas:")
        for key, value in results['metrics'].items():
            if key != 'category_analysis':
                print(f"   {key}: {value}")

        if 'category_analysis' in results['metrics']:
            print("\n🏷️  Análisis por categorías:")
            cat_analysis = results['metrics']['category_analysis']

            print("   Similitud intra-categoría:")
            for cat, stats in cat_analysis['intra_category_similarity'].items():
                print(f"     {cat}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        return True

    except Exception as e:
        print(f"❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = create_demo_semantic_heatmap()
    print(f"\n{'='*50}")
    if success:
        print("🎉 Demo completado exitosamente")
        print("💡 Revisa la carpeta 'demo_heatmap_output' para ver los mapas de calor generados")
    else:
        print("❌ Demo falló. Revisa las dependencias.")
