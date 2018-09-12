

test.py
# v1.0
## test one image
```
python test.py \
    --input /home/zhangxin/data_public/slim/flower_photos_val/daisy/5547758_eea9edfd54_n.jpg \
    --model v1.0/train_iter_1000.caffemodel \
    --config v1.0/deploy.prototxt \
    --framework caffe \
    --classes flower_label.txt \
    --mean 104 117 123 \
    --width 227 \
    --height 227
```
## batch test:
```
python test.py \
    --input val_file_list.txt \
    --model v1.0/train_iter_1000.caffemodel \
    --config v1.0/deploy.prototxt \
    --framework caffe \
    --classes flower_label.txt \
    --mean 104 117 123 \
    --width 227 \
    --height 227
```
### batch test result:
```
 [[ 43.      1.      3.      1.      2.     50.     43.      0.86 ]
 [  2.     40.      1.      5.      2.     50.     40.      0.8  ]
 [  1.      0.     44.      1.      4.     50.     44.      0.88 ]
 [  0.      1.      0.     49.      0.     50.     49.      0.98 ]
 [  1.      0.      3.      0.     46.     50.     46.      0.92 ]
 [ 47.     42.     51.     56.     54.      0.      0.      0.   ]
 [ 43.     40.     44.     49.     46.      0.      0.      0.   ]
 [  0.915   0.952   0.863   0.875   0.852 250.    222.      0.888]]
1443.3283699999986 250 5.773313479999994
```

## batch test(rgb):
```
python test.py \
    --input val_file_list.txt \
    --model v1.0/train_iter_1000.caffemodel \
    --config v1.0/deploy.prototxt \
    --framework caffe \
    --classes flower_label.txt \
    --mean 104 117 123 \
    --width 227 \
    --height 227 \
    --rgb
```
### batch test result:
```
[[ 40.      2.      3.      0.      5.     50.     40.      0.8  ]
 [  1.     38.      2.      4.      5.     50.     38.      0.76 ]
 [  1.      0.     45.      1.      3.     50.     45.      0.9  ]
 [  1.      3.      8.     28.     10.     50.     28.      0.56 ]
 [  2.      0.      3.      0.     45.     50.     45.      0.9  ]
 [ 45.     43.     61.     33.     68.      0.      0.      0.   ]
 [ 40.     38.     45.     28.     45.      0.      0.      0.   ]
 [  0.889   0.884   0.738   0.848   0.662 250.    196.      0.784]]
```


# v1.1
```
python test.py \
    --input /home/zhangxin/data_public/slim/flower_photos_val/daisy/5547758_eea9edfd54_n.jpg \
    --model v1.1/train_iter_100.caffemodel \
    --config v1.1/deploy.prototxt \
    --framework caffe \
    --classes flower_label.txt \
    --mean 104 117 123 \
    --width 227 \
    --height 227
```

## batch test:
```
python test.py \
    --input val_file_list.txt \
    --model v1.1/train_iter_100.caffemodel \
    --config v1.1/deploy.prototxt \
    --framework caffe \
    --classes flower_label.txt \
    --mean 104 117 123 \
    --width 227 \
    --height 227
```
### batch test result:
```
[[ 43.      2.      2.      1.      2.     50.     43.      0.86 ]
 [  3.     36.      1.      7.      3.     50.     36.      0.72 ]
 [  1.      0.     45.      1.      3.     50.     45.      0.9  ]
 [  0.      2.      2.     46.      0.     50.     46.      0.92 ]
 [  2.      0.      4.      0.     44.     50.     44.      0.88 ]
 [ 49.     40.     54.     55.     52.      0.      0.      0.   ]
 [ 43.     36.     45.     46.     44.      0.      0.      0.   ]
 [  0.878   0.9     0.833   0.836   0.846 250.    214.      0.856]]
693.9804119999994 250 2.7759216479999975
```


## batch test(rgb):
```
python test.py \
    --input val_file_list.txt \
    --model v1.1/train_iter_100.caffemodel \
    --config v1.1/deploy.prototxt \
    --framework caffe \
    --classes flower_label.txt \
    --mean 104 117 123 \
    --width 227 \
    --height 227 \
    --rgb
```
### batch test result:
```
[[ 42.      1.      2.      0.      5.     50.     42.      0.84 ]
 [  8.     33.      3.      0.      6.     50.     33.      0.66 ]
 [  0.      0.     46.      0.      4.     50.     46.      0.92 ]
 [  7.      6.     18.     13.      6.     50.     13.      0.26 ]
 [  2.      0.      5.      0.     43.     50.     43.      0.86 ]
 [ 59.     40.     74.     13.     64.      0.      0.      0.   ]
 [ 42.     33.     46.     13.     43.      0.      0.      0.   ]
 [  0.712   0.825   0.622   1.      0.672 250.    177.      0.708]]
```