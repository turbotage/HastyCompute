import simulate_mri as simri

simri.simulate(dirpath='D:\\4DRecon\\dat\\dat2', imagefile='images_6f.h5',
    create_crop_image=False, load_crop_image=True,
    create_enc_image=False, load_enc_image=True,
    create_nufft_of_enced_image=False, load_nufft_of_neced_image=False)