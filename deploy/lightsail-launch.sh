#!/bin/bash
# Paste into Lightsail "Create instance" -> "Add launch script" (Ubuntu 22.04/24.04).
# Runs once as root on first boot: installs the IDR replay service, pulls the
# recordings from Google Drive, and serves it on port 80. Progress log:
#   sudo tail -f /var/log/idr-setup.log
set -eux
exec > /var/log/idr-setup.log 2>&1

BRANCH=render-deploy
export DATA_FOLDER_URL="https://drive.google.com/drive/folders/1t_RECqwdEEg5rAYrYMfqriabc7RVvW0l"

apt-get update
apt-get install -y python3-venv git
cd /home/ubuntu
sudo -u ubuntu git clone --depth 1 -b "$BRANCH" https://github.com/piyushiitm/TrajectoryPredictor.git
cd TrajectoryPredictor
sudo -u ubuntu python3 -m venv venv
sudo -u ubuntu ./venv/bin/pip install -r server/requirements-render.txt

cat > /etc/systemd/system/idr.service <<UNIT
[Unit]
Description=IDR replay
After=network-online.target
[Service]
WorkingDirectory=/home/ubuntu/TrajectoryPredictor
Environment=DATA_FOLDER_URL=$DATA_FOLDER_URL
ExecStart=/home/ubuntu/TrajectoryPredictor/venv/bin/uvicorn server:app --app-dir server --host 0.0.0.0 --port 80
Restart=always
AmbientCapabilities=CAP_NET_BIND_SERVICE
User=ubuntu
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now idr        # site is up now; the server fetches Drive data in the background
echo "IDR setup done"
