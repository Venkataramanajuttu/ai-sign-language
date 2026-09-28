#!/usr/bin/env bash
set -e; NAME=signlanguage; PORT=8501
docker info >/dev/null 2>&1 || { echo "Start Docker, then re-run."; exit 1; }
docker build -t $NAME .
docker rm -f $NAME 2>/dev/null || true
docker run -d --name $NAME -p $PORT:$PORT $NAME
( sleep 6; (open "http://localhost:$PORT" 2>/dev/null || xdg-open "http://localhost:$PORT") ) &
echo "Open http://localhost:$PORT   (stop: docker rm -f $NAME)"
