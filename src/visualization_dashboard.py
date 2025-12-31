"""
Dashboard de visualización para comparar resultados de diferentes modelos de embeddings.
Incluye gráficos interactivos, tablas de comparación y análisis detallado de métricas.
"""

import json
import logging
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# Imports condicionales para visualizaciones avanzadas
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from embedding_evaluation_framework import EmbeddingModelInfo, EvaluationMetrics
except ImportError:
    # Definiciones simples para fallback
    class EvaluationMetrics:
        pass
    class EmbeddingModelInfo:
        pass

logger = logging.getLogger(__name__)


class EmbeddingVisualizationDashboard:
    """
    Dashboard para visualizar y comparar resultados de evaluación de embeddings
    """

    def __init__(self, results_dir: str = "embedding_evaluation_results"):
        """
        Inicializa el dashboard de visualización

        Args:
            results_dir: Directorio con resultados de evaluación
        """
        self.results_dir = Path(results_dir)
        self.results_data = {}
        self.model_info = {}

        # Configurar estilo de matplotlib
        plt.style.use('default')
        sns.set_palette("husl")

        # Configurar logging
        logging.basicConfig(level=logging.INFO)

        # Cargar datos si existen
        self.load_results()

    def load_results(self):
        """Carga los resultados del benchmark desde archivos JSON"""
        results_file = self.results_dir / "benchmark_results.json"
        model_info_file = self.results_dir / "model_info.json"

        if results_file.exists():
            with open(results_file, encoding='utf-8') as f:
                self.results_data = json.load(f)
            logger.info(f"✅ Resultados cargados desde {results_file}")
        else:
            logger.warning(f"⚠️  No se encontraron resultados en {results_file}")

        if model_info_file.exists():
            with open(model_info_file, encoding='utf-8') as f:
                self.model_info = json.load(f)
            logger.info(f"✅ Información de modelos cargada desde {model_info_file}")

    def create_comparison_dataframe(self) -> pd.DataFrame:
        """
        Crea un DataFrame consolidado para comparaciones

        Returns:
            DataFrame con métricas de todos los modelos
        """
        if not self.results_data:
            logger.warning("⚠️  No hay datos de resultados para crear DataFrame")
            return pd.DataFrame()

        data = []

        for model_name, metrics in self.results_data.items():
            # Métricas básicas
            row = {
                "modelo": model_name,
                "bert_score_f1": metrics.get("bert_score_f1", 0.0),
                "bert_score_precision": metrics.get("bert_score_precision", 0.0),
                "bert_score_recall": metrics.get("bert_score_recall", 0.0),
                "tiempo_consulta": metrics.get("query_time", 0.0),
                "tiempo_inicializacion": metrics.get("embedding_generation_time", 0.0),
                "consultas_exitosas": metrics.get("successful_queries", 0),
                "total_consultas": metrics.get("total_queries", 0),
                "tasa_error": metrics.get("error_rate", 0.0),
            }

            # Métricas precision@k y recall@k
            precision_at_k = metrics.get("precision_at_k", {})
            recall_at_k = metrics.get("recall_at_k", {})

            for k in [1, 3, 5, 10]:
                row[f"precision_at_{k}"] = precision_at_k.get(str(k), 0.0)
                row[f"recall_at_{k}"] = recall_at_k.get(str(k), 0.0)

            # Información del modelo
            if model_name in self.model_info:
                model_info = self.model_info[model_name]
                row["dimension_embedding"] = model_info.get("embedding_dim", 0)
                row["soporta_texto"] = model_info.get("supports_text_queries", False)
                row["soporta_audio"] = model_info.get("supports_audio_queries", False)
                row["dispositivo"] = model_info.get("device", "unknown")

            data.append(row)

        df = pd.DataFrame(data)
        return df

    def plot_bertscore_comparison(self, save_path: str | None = None):
        """
        Crea gráfico de comparación de BERTScore

        Args:
            save_path: Ruta para guardar el gráfico
        """
        df = self.create_comparison_dataframe()

        if df.empty:
            logger.warning("⚠️  No hay datos para graficar")
            return

        # Configurar figura
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Comparación de BERTScore entre Modelos', fontsize=16, fontweight='bold')

        metrics = ['bert_score_precision', 'bert_score_recall', 'bert_score_f1']
        titles = ['Precisión', 'Recall', 'F1-Score']
        colors = ['skyblue', 'lightcoral', 'lightgreen']

        for i, (metric, title, color) in enumerate(zip(metrics, titles, colors, strict=False)):
            ax = axes[i]

            # Gráfico de barras
            bars = ax.bar(df['modelo'], df[metric], color=color, alpha=0.7, edgecolor='black')

            # Añadir valores en las barras
            for bar, value in zip(bars, df[metric], strict=False):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

            ax.set_title(f'BERTScore {title}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)

            # Rotar etiquetas si es necesario
            if len(df) > 3:
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Gráfico BERTScore guardado en {save_path}")

        plt.show()

    def plot_precision_recall_curves(self, save_path: str | None = None):
        """
        Crea gráficos de curvas Precision@K y Recall@K

        Args:
            save_path: Ruta para guardar el gráfico
        """
        df = self.create_comparison_dataframe()

        if df.empty:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Métricas de Recuperación por Modelo', fontsize=16, fontweight='bold')

        k_values = [1, 3, 5, 10]
        markers = ['o', 's', '^', 'D']
        colors = sns.color_palette("husl", len(df))

        # Precision@K
        for i, (_, row) in enumerate(df.iterrows()):
            precision_values = [row[f'precision_at_{k}'] for k in k_values]
            ax1.plot(k_values, precision_values, marker=markers[i % len(markers)],
                    color=colors[i], linewidth=2, markersize=8, label=row['modelo'])

        ax1.set_title('Precision@K', fontweight='bold')
        ax1.set_xlabel('K')
        ax1.set_ylabel('Precision')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Recall@K
        for i, (_, row) in enumerate(df.iterrows()):
            recall_values = [row[f'recall_at_{k}'] for k in k_values]
            ax2.plot(k_values, recall_values, marker=markers[i % len(markers)],
                    color=colors[i], linewidth=2, markersize=8, label=row['modelo'])

        ax2.set_title('Recall@K', fontweight='bold')
        ax2.set_xlabel('K')
        ax2.set_ylabel('Recall')
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📈 Gráfico Precision/Recall guardado en {save_path}")

        plt.show()

    def plot_performance_radar(self, save_path: str | None = None):
        """
        Crea gráfico radar para comparar rendimiento general

        Args:
            save_path: Ruta para guardar el gráfico
        """
        df = self.create_comparison_dataframe()

        if df.empty:
            return

        # Métricas a incluir en el radar (normalizar a 0-1)
        metrics = {
            'BERTScore F1': 'bert_score_f1',
            'Precision@5': 'precision_at_5',
            'Recall@5': 'recall_at_5',
            'Velocidad': 'tiempo_consulta',  # Invertir: menor tiempo = mejor
            'Confiabilidad': 'tasa_error'    # Invertir: menor error = mejor
        }

        # Normalizar métricas
        df_norm = df.copy()

        # Para velocidad y error, invertir (1 - normalizado)
        if 'tiempo_consulta' in df.columns:
            max_time = df['tiempo_consulta'].max()
            df_norm['tiempo_consulta'] = 1 - (df['tiempo_consulta'] / max_time) if max_time > 0 else 1

        if 'tasa_error' in df.columns:
            df_norm['tasa_error'] = 1 - df['tasa_error']

        # Crear gráfico radar
        _fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Cerrar el círculo

        colors = sns.color_palette("husl", len(df))

        for i, (_, row) in enumerate(df_norm.iterrows()):
            values = []
            for _metric_name, col_name in metrics.items():
                value = row.get(col_name, 0.0)
                values.append(value)

            values += [values[0]]  # Cerrar el círculo

            ax.plot(angles, values, 'o-', linewidth=2, label=row['modelo'], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        # Configurar radar
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics.keys())
        ax.set_ylim(0, 1)
        ax.set_title('Comparación Multidimensional de Modelos', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"🎯 Gráfico radar guardado en {save_path}")

        plt.show()

    def plot_efficiency_comparison(self, save_path: str | None = None):
        """
        Crea gráfico de comparación de eficiencia (tiempo vs. calidad)

        Args:
            save_path: Ruta para guardar el gráfico
        """
        df = self.create_comparison_dataframe()

        if df.empty:
            return

        # Gráfico de dispersión: Tiempo vs. Calidad
        _fig, ax = plt.subplots(figsize=(10, 8))

        colors = sns.color_palette("husl", len(df))
        sizes = df.get('dimension_embedding', [512] * len(df))  # Tamaño basado en dimensión

        # Normalizar tamaños para visualización
        sizes_norm = 100 + (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 300 if sizes.max() > sizes.min() else [200] * len(sizes)

        scatter = ax.scatter(df['tiempo_consulta'], df['bert_score_f1'],
                           c=colors, s=sizes_norm, alpha=0.7, edgecolors='black')

        # Añadir etiquetas a los puntos
        for _i, (_, row) in enumerate(df.iterrows()):
            ax.annotate(row['modelo'],
                       (row['tiempo_consulta'], row['bert_score_f1']),
                       xytext=(5, 5), textcoords='offset points',
                       fontweight='bold')

        ax.set_xlabel('Tiempo de Consulta (segundos)', fontweight='bold')
        ax.set_ylabel('BERTScore F1', fontweight='bold')
        ax.set_title('Eficiencia vs. Calidad de Modelos\n(Tamaño = Dimensión de Embedding)', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Añadir líneas de referencia
        ax.axhline(y=df['bert_score_f1'].mean(), color='red', linestyle='--', alpha=0.5, label='F1 promedio')
        ax.axvline(x=df['tiempo_consulta'].mean(), color='blue', linestyle='--', alpha=0.5, label='Tiempo promedio')

        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"⚡ Gráfico de eficiencia guardado en {save_path}")

        plt.show()

    def create_summary_table(self, save_path: str | None = None):
        """
        Crea tabla resumen con todas las métricas

        Args:
            save_path: Ruta para guardar la tabla
        """
        df = self.create_comparison_dataframe()

        if df.empty:
            logger.warning("⚠️  No hay datos para crear tabla")
            return None

        # Seleccionar métricas más importantes
        summary_cols = [
            'modelo',
            'bert_score_f1',
            'precision_at_5',
            'recall_at_5',
            'tiempo_consulta',
            'consultas_exitosas',
            'total_consultas',
            'dimension_embedding'
        ]

        # Filtrar columnas existentes
        available_cols = [col for col in summary_cols if col in df.columns]
        summary_df = df[available_cols].copy()

        # Formatear números
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['bert_score_f1', 'precision_at_5', 'recall_at_5']:
                summary_df[col] = summary_df[col].round(3)
            elif col == 'tiempo_consulta':
                summary_df[col] = summary_df[col].round(4)

        # Ordenar por BERTScore F1
        if 'bert_score_f1' in summary_df.columns:
            summary_df = summary_df.sort_values('bert_score_f1', ascending=False)

        # Crear visualización de tabla
        _fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')

        # Crear tabla
        table = ax.table(cellText=summary_df.values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Colorear filas alternadas
        for i in range(len(summary_df)):
            for j in range(len(summary_df.columns)):
                if i % 2 == 0:
                    table[(i+1, j)].set_facecolor('#f0f0f0')

        # Colorear encabezados
        for j in range(len(summary_df.columns)):
            table[(0, j)].set_facecolor('#4CAF50')
            table[(0, j)].set_text_props(weight='bold', color='white')

        plt.title('Tabla Resumen de Comparación de Modelos', fontsize=16, fontweight='bold', pad=20)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📋 Tabla resumen guardada en {save_path}")

        plt.show()

        return summary_df

    def create_interactive_dashboard(self, save_path: str | None = None):
        """
        Crea dashboard interactivo con Plotly (si está disponible)

        Args:
            save_path: Ruta para guardar el dashboard HTML
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("⚠️  Plotly no disponible, creando dashboard estático")
            self.create_static_dashboard()
            return

        df = self.create_comparison_dataframe()

        if df.empty:
            logger.warning("⚠️  No hay datos para crear dashboard")
            return

        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('BERTScore Comparison', 'Precision@K vs Recall@K',
                           'Efficiency Analysis', 'Model Capabilities'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # 1. BERTScore comparison
        fig.add_trace(
            go.Bar(x=df['modelo'], y=df['bert_score_f1'],
                  name='BERTScore F1', marker_color='lightblue'),
            row=1, col=1
        )

        # 2. Precision vs Recall scatter
        fig.add_trace(
            go.Scatter(x=df['precision_at_5'], y=df['recall_at_5'],
                      mode='markers+text', text=df['modelo'],
                      textposition="top center", name='P@5 vs R@5',
                      marker={'size': 12, 'color': 'orange'}),
            row=1, col=2
        )

        # 3. Efficiency analysis
        fig.add_trace(
            go.Scatter(x=df['tiempo_consulta'], y=df['bert_score_f1'],
                      mode='markers+text', text=df['modelo'],
                      textposition="top center", name='Efficiency',
                      marker={'size': 15, 'color': 'green'}),
            row=2, col=1
        )

        # 4. Model capabilities
        capabilities = []
        for _, row in df.iterrows():
            caps = 0
            if row.get('soporta_texto', False):
                caps += 1
            if row.get('soporta_audio', False):
                caps += 1
            capabilities.append(caps)

        fig.add_trace(
            go.Bar(x=df['modelo'], y=capabilities,
                  name='Capabilities', marker_color='purple'),
            row=2, col=2
        )

        # Actualizar layout
        fig.update_layout(
            title_text="Dashboard Interactivo de Comparación de Modelos",
            showlegend=False,
            height=800
        )

        # Mostrar figura
        fig.show()

        # Guardar como HTML si se especifica
        if save_path:
            fig.write_html(save_path)
            logger.info(f"🌐 Dashboard interactivo guardado en {save_path}")

    def create_static_dashboard(self):
        """Crea dashboard estático usando matplotlib"""
        logger.info("📊 Creando dashboard estático...")

        # Crear múltiples gráficos
        self.plot_bertscore_comparison()
        self.plot_precision_recall_curves()
        self.plot_performance_radar()
        self.plot_efficiency_comparison()
        self.create_summary_table()

    def generate_full_report(self, output_dir: str | None = None):
        """
        Genera reporte completo con todas las visualizaciones

        Args:
            output_dir: Directorio de salida para el reporte
        """
        if output_dir is None:
            output_dir = self.results_dir / "visualizations"

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info(f"📊 Generando reporte completo en {output_path}")

        # Generar todos los gráficos
        self.plot_bertscore_comparison(output_path / "bertscore_comparison.png")
        self.plot_precision_recall_curves(output_path / "precision_recall_curves.png")
        self.plot_performance_radar(output_path / "performance_radar.png")
        self.plot_efficiency_comparison(output_path / "efficiency_comparison.png")

        # Tabla resumen
        summary_df = self.create_summary_table(output_path / "summary_table.png")

        # Guardar datos como CSV
        if summary_df is not None:
            summary_df.to_csv(output_path / "comparison_data.csv", index=False)
            logger.info(f"💾 Datos guardados en {output_path / 'comparison_data.csv'}")

        # Dashboard interactivo si está disponible
        if PLOTLY_AVAILABLE:
            self.create_interactive_dashboard(output_path / "interactive_dashboard.html")

        logger.info(f"✅ Reporte completo generado en {output_path}")


def main():
    """Función principal para generar visualizaciones"""
    print("📊 Dashboard de Visualización de Embeddings")
    print("=" * 50)

    # Crear dashboard
    dashboard = EmbeddingVisualizationDashboard()

    # Verificar si hay datos
    if not dashboard.results_data:
        print("⚠️  No se encontraron datos de evaluación.")
        print("💡 Ejecuta primero el benchmark de evaluación.")
        return

    # Generar reporte completo
    try:
        dashboard.generate_full_report()
        print("✅ Reporte de visualización completado")

    except Exception as e:
        logger.error(f"❌ Error generando reporte: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
