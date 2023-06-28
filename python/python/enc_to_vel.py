import sys

sys.path.append('C:\\Users\\TurboTage\\Documents\\GitHub\\pycompute')

from jinja2 import Template

from pycompute.cuda.lsqnonlin import F_GradF
from pycompute.cuda.cuda_program import CudaFunction, CudaTensor

from nlsq import F_GradF

class EncVelF(CudaFunction):
	def __init__(self):
		pass
	
	def get_device_funcid(self):
		return 'enc_to_vel_f'
	
	def get_device_code(self):
		f_temp = Template(
"""
__device__
"""
		)