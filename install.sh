apt install python3-pip -y
pip install torch hanziconv datasets requests
cd model_train_data
python3 model.py

sudo swapoff /swapfile
sudo rm /swapfile
sudo fallocate -l 500G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

