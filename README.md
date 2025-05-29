# dream
This is the public repo for DREAM accepted to ICS 2025.
Sorry for delay. I will update the Readme within the week until June 7.


```

sudo apt-get update
sudo apt-get install nvidia-driver-535
sudo apt-get update && sudo apt-get install -y cmake cython3 dh-python libsystemd-dev libudev-dev pandoc python3-docutils valgrind


sudo wget http://www.mellanox.com/downloads/ofed/MLNX_OFED-23.07-0.5.1.2/MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu22.04-x86_64.tgz
tar -xvf MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu22.04-x86_64.tgz
cd MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu22.04-x86_64/
sudo ./mlnxofedinstall


# cuda-12.2:
# ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

#check df -h to make sure the following directory has enough space:
cd /dev/shm/
sudo wget https://nrvis.com/download/data/massive/soc-twitter-mpi-sws.zip 
# http://konect.cc/files/download.tsv.twitter_mpi.tar.bz2
tar -xvjf download.tsv.twitter_mpi.tar.bz2


sudo wget https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz

cd
cd /uvm/UVM_Benchmarks/bfs
./main 3 1000 10000 /dev/shm/



# Extra storage:
sudo mkdir /mydata
sudo /usr/local/etc/emulab/mkextrafs.pl -f /mydata

# to get graph datasets:
sudo wget https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-kron.tar.gz
sudo wget https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-urand.tar.gz
sudo wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Friendster.tar.gz

tar -xvzf GAP-kron.tar.gz
tar -xvzf GAP-urand.tar.gz
tar -xvzf com-Friendster.tar.gz

# install freestanding
sudo rm -r freestanding/
cd ~/dream/include
git clone --recurse-submodules https://github.com/ogiroux/freestanding


```
