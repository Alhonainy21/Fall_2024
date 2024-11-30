pip3 install kaggle
nano ~/.bashrc
#Add the following line
export PATH=$PATH:/users/aga5h3/.local/bin
source ~/.bashrc
mkdir -p ~/.kaggle/
mv kaggle.json ~/.kaggle/
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
mkdir -p HAM10000_images
cp HAM10000_images_part_1/* HAM10000_images/
cp HAM10000_images_part_2/* HAM10000_images/
#To check all are installed:
ls -1 HAM10000_images | wc -l
