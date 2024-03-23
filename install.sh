apt install python3-pip -y
pip install tensorflow hanziconv datasets requests
cd model_train_data

sudo swapoff /swapfile
sudo rm /swapfile
sudo fallocate -l 500G /swapfile
chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

python3 model.py
