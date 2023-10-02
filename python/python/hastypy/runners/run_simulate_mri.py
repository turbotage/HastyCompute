
import runner

import hastypy.util.simulate_mri as simulate_mri

if __name__ == '__main__':
	simulate_mri.simulate(dirpath='D:\\4DRecon\\dat\\dat2', imagefile='my_framed_real_100.h5',
			create_crop_image=True, load_crop_image=False,
			create_enc_image=True, load_enc_image=False,
			create_nufft_of_enced_image=True, load_nufft_of_neced_image=False,
			nimgout=20,
			nspokes=100,
			samp_per_spoke=100,#samp_per_spoke=489,
			#method='PCVIPR', # PCVIPR, MidRandom
			method='PCVIPR',
			crop_factor=1.3,
			just_plot=False,
			also_plot=True)