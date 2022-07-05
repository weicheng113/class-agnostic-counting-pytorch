### List `tar` content
```shell
tar -tf ILSVRC2015_VID.tar | tree --fromfile . > out.txt
```

### Extract specific folder
```shell
tar -xvf ILSVRC2015_VID.tar ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/ -C /media/cwei/WD_BLACK/datasets/ILSVRC2015/
```

### Generate image crops
gen_image_crops_VID.py

###