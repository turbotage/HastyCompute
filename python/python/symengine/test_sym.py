
import symengine as sm

expr = sm.sympify("abs(x**2+3+sin(x**2), y*x**2)")

print(expr)

expr_diffx = sm.diff(expr, "x")
expr_diffy = sm.diff(expr, "y")
substs, reduced = sm.cse([expr_diffx, expr_diffy])

print(substs)
print(reduced)
