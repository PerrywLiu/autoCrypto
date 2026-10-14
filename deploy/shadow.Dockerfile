FROM autocrypto-bot:analysis-20261013t123600z
COPY src/live/shadow.py /app/src/live/shadow.py
COPY scripts/run_shadow.py /app/scripts/run_shadow.py
COPY tests/test_shadow_buffer.py /app/tests/test_shadow_buffer.py
