#!/usr/bin/env bash
# Start every lab as its own x402-protected HTTP server (ports 4101-4104). Logs in runs/lab_*.log
cd "$(dirname "$0")/.."
mkdir -p runs
for pair in US_BRFSS_2023:4101 BRAZIL_PNS_2019:4102 KOREA_KNHANES_2024:4103 US_NHANES_2017:4104; do
  lab="${pair%%:*}"; port="${pair##*:}"
  venv/bin/python -m envoy.lab_server "$lab" --port "$port" > "runs/lab_$lab.log" 2>&1 &
  echo "$lab -> http://127.0.0.1:$port  (pid $!)"
done
echo "Stop with: pkill -f envoy.lab_server"
