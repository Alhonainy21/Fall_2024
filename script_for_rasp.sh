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

screen -S tcp tcpdump -i eth0 -w M_test_cash10_3cli_lung3_c3.pcap -s 94

apt-get install unzip
apt-get install sudo
apt-get install bc -y
sudo -v ; curl https://rclone.org/install.sh | sudo bash
