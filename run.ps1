$ErrorActionPreference = "Stop"
$NAME = "signlanguage"; $PORT = 8501
docker info *> $null; if ($LASTEXITCODE -ne 0) { Write-Host "Open Docker Desktop, wait for 'Engine running', then re-run."; exit 1 }
docker build -t $NAME .
docker rm -f $NAME 2>$null | Out-Null
docker run -d --name $NAME -p "${PORT}:${PORT}" $NAME
Write-Host "Starting... opening http://localhost:$PORT"; Start-Sleep 6; Start-Process "http://localhost:$PORT"
Write-Host "Stop later:  docker rm -f $NAME"
