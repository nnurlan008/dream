```
tar xvf rdma-core_2307mlnx47.orig.tar.gz &&
cd rdma-core-2307mlnx47/ && 
dpkg-buildpackage -us -uc -Pmodule &&
cd .. &&
sudo /usr/bin/dpkg -i --force-confnew --force-confmiss 'rdma-core_2307mlnx47-1.2307050_amd64.deb' &&
sudo /usr/bin/dpkg -i --force-confnew --force-confmiss 'libibverbs1_2307mlnx47-1.2307050_amd64.deb' &&
sudo /usr/bin/dpkg -i --force-confnew --force-confmiss 'ibverbs-utils_2307mlnx47-1.2307050_amd64.deb' &&
sudo /usr/bin/dpkg -i --force-confnew --force-confmiss 'ibverbs-providers_2307mlnx47-1.2307050_amd64.deb' &&
sudo /usr/bin/dpkg -i --force-confnew --force-confmiss 'libibverbs-dev_2307mlnx47-1.2307050_amd64.deb' &&
sudo /usr/bin/dpkg -i --force-confnew --force-confmiss 'libibverbs1-dbg_2307mlnx47-1.2307050_amd64.deb' &&
sudo /usr/bin/dpkg -i --force-confnew --force-confmiss 'libibumad3_2307mlnx47-1.2307050_amd64.deb' &&
sudo /usr/bin/dpkg -i --force-confnew --force-confmiss 'libibumad-dev_2307mlnx47-1.2307050_amd64.deb' &&
sudo /usr/bin/dpkg -i --force-confnew --force-confmiss 'ibacm_2307mlnx47-1.2307050_amd64.deb' &&
sudo /usr/bin/dpkg -i --force-confnew --force-confmiss 'librdmacm1_2307mlnx47-1.2307050_amd64.deb' &&
sudo /usr/bin/dpkg -i --force-confnew --force-confmiss 'librdmacm-dev_2307mlnx47-1.2307050_amd64.deb'
```
