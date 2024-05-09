#!/bin/bash
until python3 /home/thechancher/ViT/server.py; do
    echo "Server error. Restarting..."
    sleep 5
done
