#!/usr/bin/python3

import os
import numpy as np

# initialize file
train =  open("./meta/train.json", 'w')
test = open("./meta/test.json", 'w')

train.write("{\n")
test.write("{\n")


for root, dirs, files in os.walk("./img"):
    for j,name in enumerate(dirs):
        f_path = os.path.join(root,name)
        train.write('"{}": [\n'.format(name))
        test.write('"{}": [\n'.format(name))
        for _,__, files in os.walk(f_path):
            files = np.array(files)
            np.random.shuffle(files)
            N = len(files)/10
            thresh = int(N/5)
            N = int(N)
            for i in range(N):
                path = files[i]
                if (i < thresh ):
                    if i == thresh -1:
                        test.write('"{}/{}"\n'.format(name,path))
                    else:
                        test.write('"{}/{}",\n'.format(name,path))
                elif ( i>= thresh and i != N-1):
                    train.write('"{}/{}",\n'.format(name,path))
                else:
                    train.write('"{}/{}"\n'.format(name,path))
            train.write(']')
            test.write(']')
            if j == 0:
                train.write(',')
                test.write(',')
            train.write('\n')
            test.write('\n')
test.write('}')
train.write('}')


