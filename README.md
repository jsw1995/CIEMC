# CIEMC
Color image encryption with meaningful Ciphertext based on 2D-INLM chaos and CNN-net Compressiont

Encryption and decryption can be achieved by running the mian function directly.

The path between the plaintext and the cover is modified in the following two lines of code

plaintext = cv.imread("D:\python\color com\com_and_encry\image\pepper.bmp")  
cover = cv.imread("D:\python\date\Encryption_standard_image\misc/4.2.06.tiff")

The parameter path of the compressed reconfiguration network is modified in the following code
(Note that the model parameters given here are for model) CNN-net*. Please contact directly if you want the CNN-net parameters.)

model.loaddict(root='model/vel')
