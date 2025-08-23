sudo mkdir -p $2
sudo mount -t nfs -o rw,vers=4.2,nconnect=8,rsize=1048576,wsize=1048576,noatime $1:/ $2