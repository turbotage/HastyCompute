# Memory ledger per matrix (bytes on the device for one (frame, encode) gradient, static engine) and samples per frame.
import math, json
def smooth(n):
    m=max(n,2)
    while True:
        if m%2==0:
            r=m
            for p in (2,3,5,7):
                while r%p==0: r//=p
            if r==1: return m
        m+=1
G=1e9
meta=json.load(open('/home/turbotage/Documents/GitHub/HastyCompute/data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/meta.json'))
nro,nadc,nenc=meta['n_readouts'],meta['n_adc'],meta['sequence']['n_enc']
tot=nro*nadc
print(f"scan: {nro} readouts x {nadc} ADC = {tot:.4g} samples in {meta['scan_time']} s, {nenc} encodes -> {tot/nenc:.4g} per encode")
for F in (7,20,48,67,96):
    print(f"  F={F:3d} frames: per (frame, encode) {tot/nenc/F/1e6:.3f} M, per frame (all encodes) {tot/F/1e6:.2f} M")
for m in [(192,)*3,(256,)*3,(320,)*3,(512,)*3,(640,384,320)]:
    nf=[smooth(math.ceil(1.25*n)) for n in m]
    nv=math.prod(m); ge=math.prod(nf)
    ch=ge*8
    for nc,L in ((10,10),(16,16),(10,16),(16,10)):
        fixed=nc*nv*8+L*nv*8+2*nv*8
        print(f"{m} grid {nf}: channel {ch/G:.3f} GB; nc {nc} L {L}: maps {nc*nv*8/G:.2f} + diag {L*nv*8/G:.2f} + x,xo {2*nv*8/G:.2f} = {fixed/G:.2f} GB fixed; all {nc*L} channels {nc*L*ch/G:.1f} GB; one coil {L*ch/G:.1f} GB; plane {nf[0]*nf[1]*8*nc*L/1e6:.0f} MB (all ch), per ch {nf[0]*nf[1]*8/1e6:.2f} MB")
