mkdir split_frame_train
cd split_frame_train
wget https://www.dropbox.com/s/gkpwlmzr3u6vyab/1-50.zip?dl=1 -O 1-50.zip
wget https://www.dropbox.com/s/z4yz4f3uaqs4txc/51-100.zip?dl=1 -O 51-100.zip
wget https://www.dropbox.com/s/8fzp0s23j9jxu16/101-150.zip?dl=1 -O 101-150.zip
wget https://www.dropbox.com/s/v1nr22mtcuvwle9/151-200.zip?dl=1 -O 151-200.zip
wget https://www.dropbox.com/s/l3yoc3mznvfpkg9/201-250.zip?dl=1 -O 201-250.zip
wget https://www.dropbox.com/s/t0fxl0ydmxyme64/251-300.zip?dl=1 -O 251-300.zip
wget https://www.dropbox.com/s/bxxf9f5mx784irr/301-350.zip?dl=1 -O 301-350.zip
wget https://www.dropbox.com/s/kvj5595h5e89bag/351-394.zip?dl=1 -O 351-394.zip
unzip -q "./1-50.zip"
unzip -q "./51-100.zip"
unzip -q "./101-150.zip"
unzip -q "./151-200.zip"
unzip -q "./201-250.zip"
unzip -q "./251-300.zip"
unzip -q "./301-350.zip"
unzip -q "./351-394.zip"
rm *.zip
cd ..


wget https://www.dropbox.com/s/mvdocfg88rpym0j/split_frame_test.zip?dl=1 -O split_frame_test.zip
unzip -q "./split_frame_test.zip"
rm split_frame_test.zip


mkdir mfcc
cd mfcc
wget https://www.dropbox.com/s/yg6e7vrt9xk7y8f/1-8000_mfcc.zip?dl=1 -O A.zip
wget https://www.dropbox.com/s/ajsgmwa3m8g03fe/8001-18800_mfcc.zip?dl=1 -O B.zip
wget https://www.dropbox.com/s/2fc7ofarrk7iyfv/18801-26683.zip?dl=1 -O C.zip
unzip -q A.zip
unzip -q B.zip
unzip -q C.zip
cd ..

gdown --id '11cuQu9U_LMTqjW8ux1Z6pJPxXavqbPaz' --output VIS_best.ckpt
gdown --id '14m35LeGqH6xd1Hd8Cg1fOXnTe8egiixg' --output AUD_best.ckpt
gdown --id '1UrDnmuWlc0sZGvLDM9lHK8fKPl1rTxcY' --output COMB_all_best.ckpt