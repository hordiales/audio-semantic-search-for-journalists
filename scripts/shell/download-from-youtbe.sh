mkdir data
cd data

URL="$1"

echo "Nota: Si falla recordar pasar el argumento/url entre comillas"
yt-dlp -x "$URL"

