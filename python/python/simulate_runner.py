import simulate_mri as simri

simri.simulate(dirpath='D:\\4DRecon\\dat\\dat2', imagefile='images_6f.h5',
    create_crop_image=True, load_crop_image=False,
    create_enc_image=True, load_enc_image=False,
    create_nufft_of_enced_image=True, load_nufft_of_neced_image=False,
    just_plot=False)