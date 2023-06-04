
import symengine as sm

expr = sm.sympify("x*sqrt(x**2 + y**2) + asinh(x*y)")

print(expr)

expr_diffx = sm.diff(expr, "x")
expr_diffy = sm.diff(expr, "y")
substs, reduced = sm.cse([expr_diffx, expr_diffy])

print(expr_diffx)
print(expr_diffy)

print(substs)
print(reduced)
