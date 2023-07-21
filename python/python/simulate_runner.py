import simulate_mri as simri

simri.simulate(dirpath='D:\\4DRecon\\dat\\dat2', imagefile='my_framed_20f.h5',
    create_crop_image=True, load_crop_image=False,
    create_enc_image=True, load_enc_image=False,
    create_nufft_of_enced_image=True, load_nufft_of_neced_image=False,
    nimgout=20,
    nspokes=800,
	samp_per_spoke=489,
	method='PCVIPR', # PCVIPR, MidRandom
	crop_factor=1.5,
    just_plot=False,
    also_plot=True)

