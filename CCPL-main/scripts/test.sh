python ./test.py \
--vgg models/vgg_normalised.pth \
--SCT ./artistic/sct/data/data.pkl \
--decoder ./artistic/decoder/data/data.pkl \
--content input/content/brad_pitt.jpg \
--style input/style/flower_of_life.jpg \
--testing_mode art

python test.py --content input/content/brad_pitt.jpg --style input/style/flower_of_life.jpg --decoder artistic/decoder_iter_160000.pth.tar --SCT artistic/sct_iter_160000.pth.tar --testing_mode art