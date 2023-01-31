echo "Hello World"

find idc-wsi-conversion/Data -follow -name '*.svs' -exec ./idc-wsi-conversion/gdcsvstodcm.sh '{}' ';'
