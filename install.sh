apt install python3-pip -y
pip install torch hanziconv datasets requests
cd model_train_data

sudo swapoff /swapfile
sudo rm /swapfile
sudo fallocate -l 300G /swapfile
chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo apt update
sudo apt install ufw
sudo ufw allow 12345/tcp

python3 model.py

