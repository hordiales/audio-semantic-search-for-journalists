"""
Estrategias y métricas comprehensivas para evaluar espectrogramas de audio.
Incluye métricas de calidad, características espectrales, comparación y análisis estadístico.

Uso:
    evaluator = SpectrogramEvaluator()
    metrics = evaluator.evaluate_spectrogram(spectrogram, sample_rate=16000)
    comparison = evaluator.compare_spectrograms(spec1, spec2)
"""

from dataclasses import dataclass, field
import logging
from typing import Any
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Imports condicionales
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy import signal
    from scipy.spatial.distance import cosine, euclidean
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SpectrogramMetrics:
    """Métricas completas de un espectrograma"""

    # Identificación
    name: str = "spectrogram"

    # Métricas básicas de forma
    n_frequencies: int = 0
    n_frames: int = 0
    duration_seconds: float = 0.0
    frequency_resolution_hz: float = 0.0
    time_resolution_seconds: float = 0.0

    # Métricas de energía y potencia
    total_energy: float = 0.0
    mean_energy: float = 0.0
    std_energy: float = 0.0
    max_energy: float = 0.0
    min_energy: float = 0.0
    energy_percentile_50: float = 0.0
    energy_percentile_90: float = 0.0
    energy_percentile_99: float = 0.0

    # Métricas espectrales (promediadas en el tiempo)
    spectral_centroid: float = 0.0  # Hz - centro de masa espectral
    spectral_bandwidth: float = 0.0  # Hz - ancho de banda espectral
    spectral_rolloff: float = 0.0  # Hz - frecuencia por debajo del cual está el 85% de la energía
    spectral_flatness: float = 0.0  # 0-1, más plano = más ruido
    spectral_crest: float = 0.0  # relación pico/promedio
    spectral_kurtosis: float = 0.0  # cuánto se concentra la energía
    spectral_skewness: float = 0.0  # asimetría espectral

    # Métricas temporales (por frecuencia)
    spectral_flux: float = 0.0  # tasa de cambio espectral
    spectral_contrast: float = 0.0  # contraste entre bandas espectrales
    spectral_entropy: float = 0.0  # entropía espectral (distribución de energía)

    # Métricas de distribución de energía
    energy_in_bands: dict[str, float] = field(default_factory=dict)  # energía por bandas (bajo, medio, alto)
    dominant_frequency: float = 0.0  # frecuencia dominante (Hz)
    frequency_spread: float = 0.0  # dispersión de frecuencias activas

    # Métricas de calidad
    signal_to_noise_ratio: float = 0.0  # SNR estimado
    dynamic_range_db: float = 0.0  # rango dinámico en dB
    sparsity: float = 0.0  # porcentaje de bins con energía significativa

    # Métricas estadísticas
    mean_magnitude: float = 0.0
    std_magnitude: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    entropy: float = 0.0  # entropía de Shannon

    # Métricas de variabilidad temporal
    temporal_variance: float = 0.0
    temporal_stability: float = 0.0  # 0-1, más alto = más estable

    # Detalles adicionales
    sample_rate: int = 16000
    hop_length: int | None = None
    n_fft: int | None = None


@dataclass
class SpectrogramComparison:
    """Resultados de comparación entre dos espectrogramas"""

    # Distancias/métricas de similitud
    euclidean_distance: float = 0.0
    cosine_similarity: float = 0.0
    correlation_coefficient: float = 0.0
    mse: float = 0.0  # Mean Squared Error
    mae: float = 0.0  # Mean Absolute Error

    # Comparación de características espectrales
    centroid_difference: float = 0.0
    bandwidth_difference: float = 0.0
    rolloff_difference: float = 0.0

    # Comparación de energía
    energy_ratio: float = 0.0
    energy_correlation: float = 0.0

    # Comparación temporal
    temporal_alignment_similarity: float = 0.0

    # Métricas de forma
    shape_similarity: float = 0.0  # similitud de formas (usando transformada)

    # Métricas adicionales
    kl_divergence: float = 0.0  # divergencia de Kullback-Leibler
    js_divergence: float = 0.0  # divergencia de Jensen-Shannon


class SpectrogramEvaluator:
    """
    Evaluador comprehensivo de espectrogramas con múltiples estrategias y métricas.
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 2048,
                 hop_length: int | None = None,
                 frequency_bands: dict[str, tuple[float, float]] | None = None):
        """
        Inicializa el evaluador de espectrogramas.

        Args:
            sample_rate: Frecuencia de muestreo del audio (Hz)
            n_fft: Tamaño de la ventana FFT
            hop_length: Longitud del salto entre frames (si None, usa n_fft//4)
            frequency_bands: Diccionario con bandas de frecuencia personalizadas
                           Ejemplo: {'bajo': (0, 200), 'medio': (200, 2000), 'alto': (2000, None)}
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4

        # Bandas de frecuencia por defecto (en Hz)
        if frequency_bands is None:
            nyquist = sample_rate / 2
            self.frequency_bands = {
                'sub_bass': (20, 60),
                'bass': (60, 250),
                'low_mid': (250, 500),
                'mid': (500, 2000),
                'high_mid': (2000, 4000),
                'presence': (4000, 6000),
                'brilliance': (6000, nyquist)
            }
        else:
            self.frequency_bands = frequency_bands

        if not LIBROSA_AVAILABLE:
            logger.warning("librosa no disponible, algunas métricas avanzadas pueden no funcionar")

    def evaluate_spectrogram(self,
                            spectrogram: np.ndarray,
                            frequencies: np.ndarray | None = None,
                            name: str = "spectrogram",
                            **kwargs) -> SpectrogramMetrics:
        """
        Evalúa un espectrograma y calcula todas las métricas disponibles.

        Args:
            spectrogram: Array 2D del espectrograma (freq_bins x time_frames)
                       Puede ser magnitud, potencia o dB
            frequencies: Array de frecuencias correspondientes (Hz). Si None, se calcula.
            name: Nombre identificador para el espectrograma
            **kwargs: Parámetros adicionales (sample_rate, hop_length, etc.)

        Returns:
            SpectrogramMetrics con todas las métricas calculadas
        """
        # Convertir a numpy si es necesario
        spec = np.asarray(spectrogram)
        if spec.ndim != 2:
            raise ValueError(f"El espectrograma debe ser 2D, recibido: {spec.ndim}D")

        # Actualizar parámetros si se proporcionan
        sample_rate = kwargs.get('sample_rate', self.sample_rate)
        hop_length = kwargs.get('hop_length', self.hop_length)
        n_fft = kwargs.get('n_fft', self.n_fft)

        # Calcular frecuencias si no se proporcionan
        if frequencies is None:
            frequencies = np.linspace(0, sample_rate / 2, spec.shape[0])

        # Convertir a magnitud lineal si está en dB
        magnitude = self._to_linear_magnitude(spec)

        # Calcular resolución temporal y de frecuencia
        time_resolution = hop_length / sample_rate
        frequency_resolution = sample_rate / (2 * (spec.shape[0] - 1))
        duration = spec.shape[1] * time_resolution

        # Inicializar métricas
        metrics = SpectrogramMetrics(
            name=name,
            n_frequencies=spec.shape[0],
            n_frames=spec.shape[1],
            duration_seconds=duration,
            frequency_resolution_hz=frequency_resolution,
            time_resolution_seconds=time_resolution,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft
        )

        # Calcular métricas de energía
        metrics = self._compute_energy_metrics(magnitude, metrics)

        # Calcular métricas espectrales
        metrics = self._compute_spectral_metrics(magnitude, frequencies, metrics)

        # Calcular métricas temporales
        metrics = self._compute_temporal_metrics(magnitude, metrics)

        # Calcular distribución de energía por bandas
        metrics = self._compute_frequency_band_energy(magnitude, frequencies, metrics)

        # Calcular métricas de calidad
        metrics = self._compute_quality_metrics(magnitude, metrics)

        # Calcular métricas estadísticas
        metrics = self._compute_statistical_metrics(magnitude, metrics)

        # Calcular variabilidad temporal
        metrics = self._compute_temporal_variability(magnitude, metrics)

        return metrics

    def compare_spectrograms(self,
                            spec1: np.ndarray,
                            spec2: np.ndarray,
                            frequencies1: np.ndarray | None = None,
                            frequencies2: np.ndarray | None = None,
                            name1: str = "spectrogram_1",
                            name2: str = "spectrogram_2") -> SpectrogramComparison:
        """
        Compara dos espectrogramas y calcula métricas de similitud.

        Args:
            spec1, spec2: Arrays 2D de los espectrogramas
            frequencies1, frequencies2: Arrays de frecuencias (opcional)
            name1, name2: Nombres identificadores

        Returns:
            SpectrogramComparison con métricas de comparación
        """
        spec1 = np.asarray(spec1)
        spec2 = np.asarray(spec2)

        # Normalizar a la misma forma si es necesario
        if spec1.shape != spec2.shape:
            spec1, spec2 = self._align_spectrograms(spec1, spec2)

        # Convertir a magnitud lineal
        mag1 = self._to_linear_magnitude(spec1)
        mag2 = self._to_linear_magnitude(spec2)

        comparison = SpectrogramComparison()

        # Calcular métricas base de los espectrogramas
        metrics1 = self.evaluate_spectrogram(spec1, frequencies1, name1)
        metrics2 = self.evaluate_spectrogram(spec2, frequencies2, name2)

        # Distancias vectoriales
        if SCIPY_AVAILABLE:
            comparison.euclidean_distance = float(euclidean(mag1.flatten(), mag2.flatten()))
            comparison.cosine_similarity = float(1 - cosine(mag1.flatten(), mag2.flatten()))

        # Correlación
        comparison.correlation_coefficient = float(np.corrcoef(mag1.flatten(), mag2.flatten())[0, 1])

        # Errores
        comparison.mse = float(np.mean((mag1 - mag2) ** 2))
        comparison.mae = float(np.mean(np.abs(mag1 - mag2)))

        # Comparación de características espectrales
        comparison.centroid_difference = abs(metrics1.spectral_centroid - metrics2.spectral_centroid)
        comparison.bandwidth_difference = abs(metrics1.spectral_bandwidth - metrics2.spectral_bandwidth)
        comparison.rolloff_difference = abs(metrics1.spectral_rolloff - metrics2.spectral_rolloff)

        # Comparación de energía
        total_energy1 = np.sum(mag1)
        total_energy2 = np.sum(mag2)
        comparison.energy_ratio = float(total_energy1 / total_energy2) if total_energy2 > 0 else 0.0
        comparison.energy_correlation = float(np.corrcoef(mag1.sum(axis=0), mag2.sum(axis=0))[0, 1])

        # Divergencias
        if SCIPY_AVAILABLE:
            comparison.kl_divergence = self._kl_divergence(mag1, mag2)
            comparison.js_divergence = self._js_divergence(mag1, mag2)

        # Similitud temporal (correlación cruzada de energías temporales)
        temporal_energy1 = mag1.sum(axis=0)
        temporal_energy2 = mag2.sum(axis=0)
        comparison.temporal_alignment_similarity = float(
            np.corrcoef(temporal_energy1, temporal_energy2)[0, 1]
        )

        return comparison

    def batch_evaluate(self,
                      spectrograms: list[np.ndarray],
                      names: list[str] | None = None,
                      frequencies: list[np.ndarray] | None = None) -> pd.DataFrame:
        """
        Evalúa múltiples espectrogramas y retorna un DataFrame con todas las métricas.

        Args:
            spectrograms: Lista de arrays de espectrogramas
            names: Lista de nombres (opcional)
            frequencies: Lista de arrays de frecuencias (opcional)

        Returns:
            DataFrame con una fila por espectrograma y columnas de métricas
        """
        if names is None:
            names = [f"spec_{i}" for i in range(len(spectrograms))]

        results = []
        for i, spec in enumerate(spectrograms):
            freq = frequencies[i] if frequencies and i < len(frequencies) else None
            name = names[i] if i < len(names) else f"spec_{i}"

            metrics = self.evaluate_spectrogram(spec, frequencies=freq, name=name)
            results.append(self._metrics_to_dict(metrics))

        return pd.DataFrame(results)

    def _to_linear_magnitude(self, spec: np.ndarray) -> np.ndarray:
        """Convierte espectrograma a magnitud lineal si está en dB."""
        # Detectar si está en dB (valores negativos comunes)
        if np.any(spec < 0) and np.max(spec) < 100:
            # Probablemente está en dB
            return librosa.db_to_amplitude(spec) if LIBROSA_AVAILABLE else np.power(10, spec / 20)
        return np.abs(spec)

    def _compute_energy_metrics(self, magnitude: np.ndarray, metrics: SpectrogramMetrics) -> SpectrogramMetrics:
        """Calcula métricas relacionadas con energía."""
        # Energía total y promedio
        total_energy = np.sum(magnitude ** 2)
        mean_energy = np.mean(magnitude ** 2)
        std_energy = np.std(magnitude ** 2)

        metrics.total_energy = float(total_energy)
        metrics.mean_energy = float(mean_energy)
        metrics.std_energy = float(std_energy)
        metrics.max_energy = float(np.max(magnitude ** 2))
        metrics.min_energy = float(np.min(magnitude ** 2))

        # Percentiles de energía
        energy_flat = (magnitude ** 2).flatten()
        metrics.energy_percentile_50 = float(np.percentile(energy_flat, 50))
        metrics.energy_percentile_90 = float(np.percentile(energy_flat, 90))
        metrics.energy_percentile_99 = float(np.percentile(energy_flat, 99))

        return metrics

    def _compute_spectral_metrics(self,
                                 magnitude: np.ndarray,
                                 frequencies: np.ndarray,
                                 metrics: SpectrogramMetrics) -> SpectrogramMetrics:
        """Calcula métricas espectrales promediadas en el tiempo."""
        if not LIBROSA_AVAILABLE:
            return metrics

        # Promediar a lo largo del tiempo para obtener perfil espectral promedio
        spectral_mean = np.mean(magnitude, axis=1)

        # Espectro de potencia
        power = magnitude ** 2
        spectral_power_mean = np.mean(power, axis=1)

        # Normalizar para distribuciones de probabilidad
        power_norm = spectral_power_mean / (np.sum(spectral_power_mean) + 1e-10)

        # Spectral Centroid (centro de masa espectral)
        if np.sum(spectral_power_mean) > 0:
            metrics.spectral_centroid = float(np.sum(frequencies * spectral_power_mean) / np.sum(spectral_power_mean))

        # Spectral Bandwidth (desviación estándar alrededor del centroide)
        if metrics.spectral_centroid > 0:
            spread = ((frequencies - metrics.spectral_centroid) ** 2) * spectral_power_mean
            metrics.spectral_bandwidth = float(np.sqrt(np.sum(spread) / (np.sum(spectral_power_mean) + 1e-10)))

        # Spectral Rolloff (85% de energía acumulada)
        cumsum_power = np.cumsum(spectral_power_mean)
        rolloff_threshold = 0.85 * cumsum_power[-1]
        rolloff_idx = np.where(cumsum_power >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            metrics.spectral_rolloff = float(frequencies[rolloff_idx[0]])

        # Spectral Flatness (relación media geométrica / media aritmética)
        # Más plano = más ruido
        spectral_power_mean_pos = spectral_power_mean + 1e-10  # evitar log(0)
        geometric_mean = np.exp(np.mean(np.log(spectral_power_mean_pos)))
        arithmetic_mean = np.mean(spectral_power_mean_pos)
        metrics.spectral_flatness = float(geometric_mean / arithmetic_mean)

        # Spectral Crest (pico / promedio)
        if arithmetic_mean > 0:
            metrics.spectral_crest = float(np.max(spectral_power_mean) / arithmetic_mean)

        # Spectral Kurtosis y Skewness
        if np.std(power_norm) > 0:
            metrics.spectral_kurtosis = float(self._kurtosis(power_norm))
            metrics.spectral_skewness = float(self._skewness(power_norm))

        # Frecuencia dominante
        dominant_idx = np.argmax(spectral_power_mean)
        metrics.dominant_frequency = float(frequencies[dominant_idx])

        # Dispersión de frecuencias activas
        # Frecuencias que contienen el 90% de la energía
        cumsum_norm = np.cumsum(np.sort(power_norm)[::-1])
        active_freqs_idx = np.where(cumsum_norm <= 0.9)[0]
        if len(active_freqs_idx) > 0:
            metrics.frequency_spread = float(len(active_freqs_idx) * (frequencies[1] - frequencies[0]))

        return metrics

    def _compute_temporal_metrics(self,
                                 magnitude: np.ndarray,
                                 metrics: SpectrogramMetrics) -> SpectrogramMetrics:
        """Calcula métricas temporales."""
        if not LIBROSA_AVAILABLE:
            return metrics

        # Spectral Flux (tasa de cambio espectral entre frames consecutivos)
        # Diferencia entre frames consecutivos
        diff = np.diff(magnitude, axis=1)
        # Suma de diferencias positivas al cuadrado
        spectral_flux = np.sum(np.maximum(diff, 0) ** 2, axis=0)
        metrics.spectral_flux = float(np.mean(spectral_flux))

        # Spectral Contrast (contraste entre bandas espectrales)
        # Dividir en bandas y calcular contraste
        n_bands = 6
        band_size = magnitude.shape[0] // n_bands
        contrasts = []
        for i in range(n_bands):
            band_start = i * band_size
            band_end = (i + 1) * band_size if i < n_bands - 1 else magnitude.shape[0]
            band = magnitude[band_start:band_end, :]
            # Contraste = diferencia entre percentiles alto y bajo
            high_percentile = np.percentile(band, 90)
            low_percentile = np.percentile(band, 10)
            contrast = high_percentile - low_percentile
            contrasts.append(contrast)
        metrics.spectral_contrast = float(np.mean(contrasts))

        # Spectral Entropy (distribución de energía)
        # Entropía de Shannon del espectro promedio
        spectral_mean = np.mean(magnitude, axis=1)
        power_norm = (spectral_mean ** 2) / (np.sum(spectral_mean ** 2) + 1e-10)
        power_norm = power_norm + 1e-10  # evitar log(0)
        if SCIPY_AVAILABLE:
            metrics.spectral_entropy = float(entropy(power_norm))
        else:
            # Entropía manual
            metrics.spectral_entropy = float(-np.sum(power_norm * np.log2(power_norm)))

        return metrics

    def _compute_frequency_band_energy(self,
                                      magnitude: np.ndarray,
                                      frequencies: np.ndarray,
                                      metrics: SpectrogramMetrics) -> SpectrogramMetrics:
        """Calcula distribución de energía por bandas de frecuencia."""
        power = magnitude ** 2
        total_power = np.sum(power)

        if total_power == 0:
            return metrics

        energy_by_band = {}
        for band_name, (freq_low, freq_high) in self.frequency_bands.items():
            # Encontrar índices de frecuencia correspondientes
            if freq_high is None:
                freq_high = frequencies[-1]

            freq_mask = (frequencies >= freq_low) & (frequencies <= freq_high)
            if np.any(freq_mask):
                band_power = np.sum(power[freq_mask, :])
                energy_by_band[band_name] = float(band_power / total_power)
            else:
                energy_by_band[band_name] = 0.0

        metrics.energy_in_bands = energy_by_band
        return metrics

    def _compute_quality_metrics(self,
                                magnitude: np.ndarray,
                                metrics: SpectrogramMetrics) -> SpectrogramMetrics:
        """Calcula métricas de calidad del espectrograma."""
        power = magnitude ** 2

        # SNR estimado (asumiendo que el ruido está en los bins menos energéticos)
        # Método simple: relación entre energía del 90% superior vs 10% inferior
        power_flat = power.flatten()
        sorted_power = np.sort(power_flat)
        n_samples = len(sorted_power)

        if n_samples > 0:
            signal_power = np.sum(sorted_power[int(0.1 * n_samples):])
            noise_power = np.sum(sorted_power[:int(0.1 * n_samples)])
            if noise_power > 0:
                metrics.signal_to_noise_ratio = float(10 * np.log10(signal_power / noise_power))

            # Rango dinámico (en dB)
            max_power = np.max(power_flat)
            min_power = np.max(sorted_power[int(0.99 * n_samples):])  # percentil 99 para evitar outliers
            if min_power > 0:
                metrics.dynamic_range_db = float(10 * np.log10(max_power / min_power))

        # Sparsity (porcentaje de bins con energía significativa)
        # Definir umbral como percentil 50
        threshold = np.percentile(power_flat, 50)
        significant_bins = np.sum(power > threshold)
        total_bins = power.size
        metrics.sparsity = float(significant_bins / total_bins) if total_bins > 0 else 0.0

        return metrics

    def _compute_statistical_metrics(self,
                                    magnitude: np.ndarray,
                                    metrics: SpectrogramMetrics) -> SpectrogramMetrics:
        """Calcula métricas estadísticas."""
        mag_flat = magnitude.flatten()

        metrics.mean_magnitude = float(np.mean(mag_flat))
        metrics.std_magnitude = float(np.std(mag_flat))

        # Skewness y Kurtosis
        if metrics.std_magnitude > 0:
            metrics.skewness = float(self._skewness(mag_flat))
            metrics.kurtosis = float(self._kurtosis(mag_flat))

        # Entropía de Shannon
        # Normalizar para distribución de probabilidad
        mag_norm = (mag_flat - np.min(mag_flat)) / (np.max(mag_flat) - np.min(mag_flat) + 1e-10)
        mag_norm = mag_norm + 1e-10
        if SCIPY_AVAILABLE:
            metrics.entropy = float(entropy(mag_norm))
        else:
            metrics.entropy = float(-np.sum(mag_norm * np.log2(mag_norm)))

        return metrics

    def _compute_temporal_variability(self,
                                     magnitude: np.ndarray,
                                     metrics: SpectrogramMetrics) -> SpectrogramMetrics:
        """Calcula métricas de variabilidad temporal."""
        # Varianza temporal (promedio de varianza de cada frecuencia a lo largo del tiempo)
        temporal_variance = np.mean(np.var(magnitude, axis=1))
        metrics.temporal_variance = float(temporal_variance)

        # Estabilidad temporal (inverso del coeficiente de variación)
        temporal_mean = np.mean(magnitude, axis=1)
        temporal_std = np.std(magnitude, axis=1)
        cv = temporal_std / (temporal_mean + 1e-10)  # coeficiente de variación
        stability = 1 / (1 + np.mean(cv))  # normalizar a 0-1
        metrics.temporal_stability = float(stability)

        return metrics

    def _align_spectrograms(self, spec1: np.ndarray, spec2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Alinea dos espectrogramas a la misma forma."""
        # Usar la forma más pequeña y recortar
        min_freq = min(spec1.shape[0], spec2.shape[0])
        min_time = min(spec1.shape[1], spec2.shape[1])

        spec1_aligned = spec1[:min_freq, :min_time]
        spec2_aligned = spec2[:min_freq, :min_time]

        return spec1_aligned, spec2_aligned

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calcula divergencia de Kullback-Leibler entre dos distribuciones."""
        # Normalizar a distribuciones de probabilidad
        p_norm = (p.flatten() + 1e-10) / (np.sum(p) + 1e-10)
        q_norm = (q.flatten() + 1e-10) / (np.sum(q) + 1e-10)

        kl = np.sum(p_norm * np.log(p_norm / q_norm))
        return float(kl)

    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calcula divergencia de Jensen-Shannon."""
        # Normalizar
        p_norm = (p.flatten() + 1e-10) / (np.sum(p) + 1e-10)
        q_norm = (q.flatten() + 1e-10) / (np.sum(q) + 1e-10)

        # Promedio
        m = 0.5 * (p_norm + q_norm)

        # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        js = 0.5 * np.sum(p_norm * np.log(p_norm / m)) + 0.5 * np.sum(q_norm * np.log(q_norm / m))
        return float(js)

    def _skewness(self, data: np.ndarray) -> float:
        """Calcula skewness (asimetría)."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = len(data)
        skew = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
        return float(skew)

    def _kurtosis(self, data: np.ndarray) -> float:
        """Calcula kurtosis (curtosis)."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = len(data)
        kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        return float(kurt)

    def _metrics_to_dict(self, metrics: SpectrogramMetrics) -> dict[str, Any]:
        """Convierte SpectrogramMetrics a diccionario para DataFrame."""
        result = {}
        for field_name, field_value in metrics.__dict__.items():
            if isinstance(field_value, dict):
                # Expandir diccionarios (como energy_in_bands)
                for key, val in field_value.items():
                    result[f"{field_name}_{key}"] = val
            else:
                result[field_name] = field_value
        return result

    def visualize_metrics_comparison(self,
                                   metrics_list: list[SpectrogramMetrics],
                                   output_path: str | None = None) -> None:
        """Visualiza comparación de métricas entre múltiples espectrogramas."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib no disponible, no se puede visualizar")
            return

        if len(metrics_list) == 0:
            logger.warning("Lista de métricas vacía")
            return

        # Convertir a DataFrame
        df = pd.DataFrame([self._metrics_to_dict(m) for m in metrics_list])

        # Crear visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Métricas de Espectrogramas', fontsize=16)

        # 1. Métricas espectrales principales
        spectral_cols = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness']
        available_cols = [c for c in spectral_cols if c in df.columns]
        if available_cols:
            df[available_cols].plot(kind='bar', ax=axes[0, 0], legend=True)
            axes[0, 0].set_title('Métricas Espectrales')
            axes[0, 0].set_ylabel('Valor')
            axes[0, 0].set_xticklabels(df['name'] if 'name' in df.columns else range(len(df)), rotation=45)

        # 2. Distribución de energía por bandas
        band_cols = [c for c in df.columns if c.startswith('energy_in_bands_')]
        if band_cols:
            band_data = df[band_cols].T
            band_data.columns = df['name'] if 'name' in df.columns else range(len(df))
            band_data.plot(kind='bar', ax=axes[0, 1], legend=True)
            axes[0, 1].set_title('Distribución de Energía por Bandas')
            axes[0, 1].set_ylabel('Proporción de Energía')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Métricas de calidad
        quality_cols = ['signal_to_noise_ratio', 'dynamic_range_db', 'sparsity']
        available_quality = [c for c in quality_cols if c in df.columns]
        if available_quality:
            df[available_quality].plot(kind='bar', ax=axes[1, 0], legend=True)
            axes[1, 0].set_title('Métricas de Calidad')
            axes[1, 0].set_ylabel('Valor')
            axes[1, 0].set_xticklabels(df['name'] if 'name' in df.columns else range(len(df)), rotation=45)

        # 4. Métricas temporales
        temporal_cols = ['spectral_flux', 'spectral_contrast', 'temporal_stability']
        available_temporal = [c for c in temporal_cols if c in df.columns]
        if available_temporal:
            df[available_temporal].plot(kind='bar', ax=axes[1, 1], legend=True)
            axes[1, 1].set_title('Métricas Temporales')
            axes[1, 1].set_ylabel('Valor')
            axes[1, 1].set_xticklabels(df['name'] if 'name' in df.columns else range(len(df)), rotation=45)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualización guardada en: {output_path}")
        else:
            plt.show()

        plt.close()


def example_usage():
    """Ejemplo de uso del evaluador de espectrogramas."""
    print("Ejemplo de uso de SpectrogramEvaluator")
    print("=" * 50)

    # Crear evaluador
    evaluator = SpectrogramEvaluator(sample_rate=16000)

    # Ejemplo: crear espectrogramas sintéticos para demostración
    if LIBROSA_AVAILABLE:
        # Generar audio sintético
        duration = 2.0
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))

        # Audio con tono puro
        audio1 = np.sin(2 * np.pi * 440 * t)  # La4

        # Audio con ruido
        audio2 = np.random.randn(int(sr * duration)) * 0.5

        # Generar espectrogramas
        spec1 = librosa.stft(audio1, n_fft=2048, hop_length=512)
        spec1_mag = np.abs(spec1)

        spec2 = librosa.stft(audio2, n_fft=2048, hop_length=512)
        spec2_mag = np.abs(spec2)

        # Evaluar
        print("\n1. Evaluando espectrograma 1 (tono puro):")
        metrics1 = evaluator.evaluate_spectrogram(spec1_mag, name="tono_puro")
        print(f"   Spectral Centroid: {metrics1.spectral_centroid:.2f} Hz")
        print(f"   Spectral Bandwidth: {metrics1.spectral_bandwidth:.2f} Hz")
        print(f"   Spectral Flatness: {metrics1.spectral_flatness:.4f}")
        print(f"   SNR: {metrics1.signal_to_noise_ratio:.2f} dB")

        print("\n2. Evaluando espectrograma 2 (ruido):")
        metrics2 = evaluator.evaluate_spectrogram(spec2_mag, name="ruido")
        print(f"   Spectral Centroid: {metrics2.spectral_centroid:.2f} Hz")
        print(f"   Spectral Bandwidth: {metrics2.spectral_bandwidth:.2f} Hz")
        print(f"   Spectral Flatness: {metrics2.spectral_flatness:.4f}")
        print(f"   SNR: {metrics2.signal_to_noise_ratio:.2f} dB")

        print("\n3. Comparando espectrogramas:")
        comparison = evaluator.compare_spectrograms(spec1_mag, spec2_mag)
        print(f"   Cosine Similarity: {comparison.cosine_similarity:.4f}")
        print(f"   MSE: {comparison.mse:.6f}")
        print(f"   KL Divergence: {comparison.kl_divergence:.4f}")

        print("\n4. Evaluación batch:")
        df = evaluator.batch_evaluate([spec1_mag, spec2_mag], names=["tono", "ruido"])
        print(df[['name', 'spectral_centroid', 'spectral_flatness', 'signal_to_noise_ratio']])
    else:
        print("librosa no disponible, no se puede ejecutar el ejemplo")


if __name__ == "__main__":
    example_usage()
