
import symengine as se
from symengine import sympify, cse, ccode, sympify, Basic

from hashlib import md5

from jinja2 import Template

class CCodePrinter:

    def doprint(self, expr, assign_to=None):
        if not isinstance(assign_to, (Basic, type(None), str)):
            raise TypeError("{} cannot assign to object of type {}".format(
                    type(self).__name__, type(assign_to)))

        expr = sympify(expr)
        if not assign_to:
            if expr.is_Matrix:
                raise RuntimeError("Matrices need a assign_to parameter")
            return ccode(expr)

        assign_to = str(assign_to)
        if not expr.is_Matrix:
            return f"{assign_to} = {ccode(expr)};"

        code_lines = []
        for i, element in enumerate(expr):
            code_line = f'{assign_to}[{i}] = {element};'
            code_lines.append(code_line)
        return '\n'.join(code_lines)

class CUDAPrinter:

	def __init__(self):
		self.cprinter = CCodePrinter()

	def tcs_f(self, sym_expr):
		sstr = self.cprinter.doprint(sym_expr)
		sstr = sstr.replace('exp(', 'expf(')
		sstr = sstr.replace('cos(', 'cosf(')
		sstr = sstr.replace('sin(', 'sinf(')
		sstr = sstr.replace('tan(', 'tanf(')
		sstr = sstr.replace('acos(', 'acosf(')
		sstr = sstr.replace('asin(', 'asinf(')
		sstr = sstr.replace('atan(', 'atanf(')
		sstr = sstr.replace('cosh(', 'coshf(')
		sstr = sstr.replace('sinh(', 'sinhf(')
		sstr = sstr.replace('tanh(', 'tanhf(')
		sstr = sstr.replace('acosh(', 'acoshf(')
		sstr = sstr.replace('asinh(', 'asinhf(')
		sstr = sstr.replace('atanh(', 'atanhf(')
		sstr = sstr.replace('sqrt(', 'sqrtf(')
		sstr = sstr.replace('rsqrt', 'rsqrtf')
		sstr = sstr.replace('pow(', 'powf(')
		return sstr

	def tcs_d(self, sym_expr):
		sstr = self.cprinter.doprint(sym_expr)
		return sstr

def expr_hash(expr, len):
    return md5(expr.encode()).hexdigest()[1:len+1]

def res(expr, pars, consts):
	resexpr = sympify(expr)
	substs, reduced = cse([resexpr])
	return (substs, reduced)

def jac(expr, pars, consts):
	resexpr = sympify(expr)

	exprs = []
	for e in pars:
		exprs.append(resexpr.diff(e))

	substs, reduced = cse(exprs)
	return (substs, reduced)

def hes(expr, pars, consts):
	resexpr = sympify(expr)

	exprs = []
	for i in range(0,len(pars)):
		for j in range(0,i+1):
			exprs.append(resexpr.diff(pars[i]).diff(pars[j]))

	substs, reduced = cse(exprs)
	return (substs, reduced)

def res_jac(expr, pars, consts):
	resexpr = sympify(expr)

	exprs = [resexpr]
	for e in pars:
		exprs.append(resexpr.diff(e))

	substs, reduced = cse(exprs)
	return (substs, reduced)

def res_jac_hes(expr, pars, consts):
	resexpr = sympify(expr)

	exprs = [resexpr]
	for e in pars:
		exprs.append(resexpr.diff(e))

	for i in range(0,len(pars)):
		for j in range(0,i+1):
			exprs.append(resexpr.diff(pars[i]).diff(pars[j]))

	substs, reduced = cse(exprs)
	return (substs,  reduced)

