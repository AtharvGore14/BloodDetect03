web: gunicorn -w 4 --timeout 120 --bind 0.0.0.0:$PORT --keep-alive 5 --graceful-timeout 120 --max-requests 1000 --max-requests-jitter 50 --worker-class sync --worker-connections 1000 --backlog 2048 --log-level info --access-logfile - --error-logfile - --capture-output --enable-stdio-inheritance app:app


