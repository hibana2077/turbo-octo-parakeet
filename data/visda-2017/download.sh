wget http://csr.bu.edu/ftp/visda17/clf/train.tar
tar xvf train.tar

wget http://csr.bu.edu/ftp/visda17/clf/validation.tar
tar xvf validation.tar  

wget http://csr.bu.edu/ftp/visda17/clf/test.tar
tar xvf test.tar

wget https://raw.githubusercontent.com/VisionLearningGroup/taskcv-2017-public/master/classification/data/image_list.txt

rm *.tar

echo "Done."

mv train/image_list.txt train/train_list.txt
mv validation/image_list.txt validation/validation_list.txt

sed -i 's#^#data/visda-2017/train/#' train/train_list.txt
sed -i 's#^#data/visda-2017/validation/#' validation/validation_list.txt