python src/simple_dataset_pipeline.py -i data/ -o ./dataset

python src/add_sentiment_to_dataset.py ./dataset
#python src/add_real_yamnet_to_dataset.py ./dataset --overwrite

# disabled by now
python src/detect_audio_events.py ./dataset --overwrite
