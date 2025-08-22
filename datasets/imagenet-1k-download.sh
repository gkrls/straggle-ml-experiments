TRAIN_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
VAL_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
DEVKIT_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"


# Accept target directory as $1, default to current directory if not provided
TARGET_DIR="${1:-.}"
mkdir -p "$TARGET_DIR"

wget -c -O "$TARGET_DIR"/ILSVRC2012_img_train.tar "$TRAIN_URL"
wget -c -O "$TARGET_DIR"/ILSVRC2012_img_val.tar "$VAL_URL"
wget -c -O "$TARGET_DIR"/ILSVRC2012_devkit_t12.tar.gz "$DEVKIT_URL"