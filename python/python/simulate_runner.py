import simulate_mri as simri

simri.simulate(dirpath='D:\\4DRecon\\dat\\dat2', imagefile='images_6f.h5',
    create_crop_image=False, load_crop_image=True,
    create_enc_image=False, load_enc_image=True,
    create_nufft_of_enced_image=True, load_nufft_of_neced_image=False,
    nspokes=510,
	samp_per_spoke=489,
	method='PCVIPR', # PCVIPR, MidRandom
	crop_factor=1.5,
    just_plot=False)

