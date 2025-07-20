conda create -n semanticsearch
conda activate UP-semanticsearch-clean

### Instalación full

```bash
# 1. Instala dependencias del sistema (macOS) o equivalente en otros sistmas operativos
brew install ffmpeg

# 2. Instala dependencias Python (mínimas)
pip install -r requirements-minimal.txt

# 3. O instalación completa (incluye TensorFlow)
pip install -r requirements.txt
```
### Opción: Instalación Asistida

```bash
# Ejecuta el script de instalación interactivo
python install.py
```

### Verificar Instalación

```bash
# Test rápido
python -c "import whisper, sentence_transformers, streamlit; print('✅ Instalación OK')"

# Test completo
python example_usage.py
```

### Problemas de Instalación

Si encuentras errores, consulta [TROUBLESHOOTING.md](TROUBLESHOOTING.md) para soluciones detalladas.