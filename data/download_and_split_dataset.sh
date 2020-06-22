ARCHIVE_NAME=lmd.tar.gz
DOWNLOAD_DIR=lmd_full/

# download lmd_full
DATASET_URL=http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
# or download the smaller clean midi dataset for testing
# DATASET_URL=http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz

wget $DATASET_URL -O $ARCHIVE_NAME

# unzip archive 
mkdir $DOWNLOAD_DIR
tar -xvzf $ARCHIVE_NAME -C $DOWNLOAD_DIR

# split dataset
python split_dataset.py $DOWNLOAD_DIR

# clean up
rm $ARCHIVE_NAME
rm -rf $DOWNLOAD_DIR