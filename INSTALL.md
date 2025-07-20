# Crear entorno virtual

    conda create -n AudioSemanticSearch python=3.11.13 -y
    conda activate AudioSemanticSearch


    ./quick_install.sh 

### Instalación de herramientas extra

```bash
# 1. Instala dependencias del sistema (macOS) o equivalente en otros sistmas operativos
brew install ffmpeg

### Verificar Instalación

```bash
# Test rápido
python -c "import whisper, sentence_transformers, streamlit; print('✅ Instalación OK')"

# Test completo
python example_usage.py
```

### Problemas de Instalación

Si encuentras errores, consulta [TROUBLESHOOTING.md](TROUBLESHOOTING.md) para soluciones detalladas.