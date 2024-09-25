sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
git clone https://github.com/lukechilds/dockerpi.git
cd dockerpi
docker run -it lukechilds/dockerpi
