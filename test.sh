#!/usr/bin/env bash
# Test the call pipeline receiver
# Usage: ./test.sh <server-url> <token>
# Example: ./test.sh http://192.168.1.100:8443 my-secret-token

SERVER="${1:-http://localhost:8443}"
TOKEN="${2:-CHANGE_ME_TO_A_RANDOM_STRING_64_CHARS_HEX}"
AUDIO="${3:-test-call.wav}"

if [ ! -f "$AUDIO" ]; then
    echo "Generating test audio file..."
    ffmpeg -y -f lavfi -i "sine=frequency=440:duration=5" -ar 16000 -ac 1 "$AUDIO" 2>/dev/null || {
        echo "No ffmpeg and no test-call.wav found. Creating a minimal WAV header..."
        # Create a minimal valid WAV file (silence)
        python3 -c "
import struct, math
sr = 16000
dur = 5
samples = [int(32767 * math.sin(2 * math.pi * 440 * t / sr)) for t in range(sr * dur)]
with open('$AUDIO', 'wb') as f:
    data_len = len(samples) * 2
    f.write(b'RIFF')
    f.write(struct.pack('<I', 36 + data_len))
    f.write(b'WAVE')
    f.write(b'fmt ')
    f.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16))
    f.write(b'data')
    f.write(struct.pack('<I', data_len))
    for s in samples:
        f.write(struct.pack('<h', s))
" 2>/dev/null
    }
fi

echo "Sending $AUDIO to $SERVER/recording/$TOKEN ..."

curl -v -X POST "$SERVER/recording/$TOKEN" \
    -F "audio=@$AUDIO" \
    -F "caller=0412345678" \
    -F "duration=42"

echo ""
echo "Done."
