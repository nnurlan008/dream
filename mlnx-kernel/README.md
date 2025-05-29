mlnx ofed kernel build and install commands:
First, `use sudo /etc/init.d/nv_peer_mem stop`
	   `sudo /etc/init.d/openibd stop`

```
cd Desktop/mlnx-ofed-kernel/
sudo /etc/init.d/nv_peer_mem stop
sudo /etc/init.d/openibd stop
tar xvf mlnx-ofed-kernel_23.07.orig.tar.gz &&
cd mlnx-ofed-kernel-23.07/ &&
dpkg-buildpackage -us -uc -Pmodule && 
cd .. &&
sudo /usr/bin/dpkg -i --force-confnew --force-confmiss 'mlnx-ofed-kernel-utils_23.07-OFED.23.07.0.5.1.1_amd64.deb' &&
sudo /usr/bin/dpkg -i --force-confnew --force-confmiss 'mlnx-ofed-kernel-dkms_23.07-OFED.23.07.0.5.1.1_all.deb'
```
