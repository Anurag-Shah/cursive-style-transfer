import shutil, random, os
dirpath = 'Cursive'
destDirectory = 'tc'

filenames = random.sample(os.listdir(dirpath), 600)
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    shutil.copyfile(srcpath, destDirectory + "/" + fname)