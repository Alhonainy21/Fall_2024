sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
git clone https://github.com/lukechilds/dockerpi.git
cd dockerpi
docker run -it lukechilds/dockerpi

wget https://cdimage.debian.org/cdimage/archive/11.7.0/armhf/iso-cd/debian-11.7.0-armhf-netinst.iso


sudo apt update
sudo apt install qemu-system-arm

apt-get install unzip
apt-get install sudo
sudo -v ; curl https://rclone.org/install.sh | sudo bash
