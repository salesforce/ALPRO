for i in $(lsof /dev/nvidia* | grep python  | awk '{print $2}' | sort -u); do kill -9 $i; done



