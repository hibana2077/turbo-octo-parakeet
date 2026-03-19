#!/usr/bin/env bash

# Add the dataset prefix to the txt files
sed -i 's#^#data/DomainNet/#' clipart_train.txt
sed -i 's#^#data/DomainNet/#' clipart_test.txt
sed -i 's#^#data/DomainNet/#' infograph_train.txt
sed -i 's#^#data/DomainNet/#' infograph_test.txt
sed -i 's#^#data/DomainNet/#' painting_train.txt
sed -i 's#^#data/DomainNet/#' painting_test.txt
sed -i 's#^#data/DomainNet/#' quickdraw_train.txt
sed -i 's#^#data/DomainNet/#' quickdraw_test.txt
sed -i 's#^#data/DomainNet/#' real_train.txt
sed -i 's#^#data/DomainNet/#' real_test.txt
sed -i 's#^#data/DomainNet/#' sketch_train.txt
sed -i 's#^#data/DomainNet/#' sketch_test.txt

echo "Done."