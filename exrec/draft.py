import numpy as np

from exrec.channels import *
from exrec.codes import *
from exrec.measurements import *
from exrec.noises import *
from exrec.decoder import *
from exrec.knill import *
from exrec.hybrid import *
from exrec.extended_gadget import *
from exrec.optimizer import *
from exrec.helpers import *
from exrec.constants import *

import time

# scheme
scheme = HYBRID
recovery = DIRECT
decoder = FAST

# code params
N = 2
alpha_data = 3.10
M = 1
alpha_anc = ALPHA_MAX

# measurement params
offset_data = -0.3
offset_anc = 0

# noise params
gamma = 0.009
gamma_phi = 0
eta = 1

code_params = [N, alpha_data, M, alpha_anc]
meas_params = [offset_data, offset_anc]
noise_params = [gamma, gamma_phi, eta]

data = CatCode(N=N, r=0, alpha=alpha_data, fockdim=DIM)
anc = CatCode(N=M, r=0, alpha=alpha_anc, fockdim=DIM)

gamma_wait = eta * gamma
gamma_phi_wait = eta * gamma_phi
base_noise = BaseNoise(scheme, gamma, gamma_phi, gamma_wait, gamma_phi_wait, mod=2 * N * M)
benchmark = BenchMark(noise_params)

# if scheme == KNILL:
#     meas_data = LogicalMeasurement(2 * N, DIM, -np.pi / (2 * N), offset_data)
#     meas_anc = LogicalMeasurement(2 * N, DIM, -np.pi / (2 * M), offset_anc)
#     meas_data.noisy([base_noise.loss_meas, base_noise.dephasing_meas])
#     meas_anc.noisy([base_noise.loss_meas, base_noise.dephasing_meas])
#
#     ec = KnillEC(data=data, anc=anc, meas_data=meas_data, meas_anc=meas_anc,
#                  gamma=gamma, gamma_phi=gamma_phi, loss_in=base_noise.loss_wait,
#                  dephasing_in=base_noise.dephasing_wait, recovery=recovery)
if scheme == HYBRID:
    meas_data = LogicalMeasurement(2 * N, DIM, -np.pi / (2 * N), offset_data)
    meas_anc = WedgeMeasurement(M * N, DIM, -np.pi / (M * N), offset_anc)
    meas_data.noisy([base_noise.loss_meas_data, base_noise.dephasing_meas_data])
    meas_anc.noisy([base_noise.loss_meas_anc, base_noise.dephasing_meas_anc])

    ec = HybridEC(data=data, anc=anc, meas_data=meas_data, meas_anc=meas_anc,
                  gamma=gamma, gamma_phi=gamma_phi, loss_in=base_noise.loss_wait,
                  dephasing_in=base_noise.dephasing_wait, recovery=recovery)
print("Init exrec...")
st = time.time()
exrec = ExtendedGadget(scheme=scheme, code_params=code_params, meas_params=meas_params,
                       noise_params=noise_params, recovery=recovery, decoder=decoder)
print("Done init exrec...", time.time() - st)
st = time.time()
print(exrec.get_infidelity(), time.time()-st)
print(exrec.get_infidelity_leading_ec())
print(exrec.get_infidelity_trailing_ec())
print(benchmark.get_infidelity())
exit()
# decoder = SDPDecoder(data, base_noise.loss, base_noise.phase)
# decoder = FastDecoder(data, base_noise.loss)
decoder = FrontDecoder(data, base_noise.loss)

cmat = ec.run(gmat=None)
dmat = ec.recovery(cmat)
out = decoder.decode(dmat)
infid = average_infidelity(out.channel_matrix)
infid_benchmark = benchmark.get_infidelity()
print(infid, infid_benchmark)
exit()


def f(params):
    alpha0, offset0, gamma_wait0 = params
    code = CatCode(N=N, r=0, alpha=alpha0, fockdim=DIM)
    ec.update_alpha(data=code, anc=code)
    ec.update_in_noise(loss_in=base_noise.loss_wait, dephasing_in=base_noise.dephasing_wait)
    sdp_decoder.update_code(code)
    base_noise.update_wait_noise(gamma_wait0, gamma_phi_wait=0)

    meas_data = LogicalMeasurement(2 * data.N, data.dim, -np.pi / (2 * data.N), offset_data)
    meas_anc = LogicalMeasurement(2 * anc.N, anc.dim, -np.pi / (2 * anc.N), offset_anc)
    meas_data.noisy([base_noise.loss_meas, base_noise.dephasing_meas])
    meas_anc.noisy([base_noise.loss_meas, base_noise.dephasing_meas])
    ec.update_measurement(meas_data, meas_anc)

    benchmark.update_noise([gamma, gamma_phi, gamma_wait0 / gamma])
    dmat = ec.run(gmat=None)
    out = sdp_decoder.decode(dmat)
    infid = average_infidelity(out.channel_matrix)
    infid_benchmark = benchmark.infidelity()
    print(params, infid, infid_benchmark)
    return infid - infid_benchmark


def g(params):
    alpha0, offset0, gamma_wait0 = params
    exrec.update_alpha([alpha0, alpha0])
    exrec.update_measurement([offset0, offset0])
    exrec.update_wait_noise(eta=gamma_wait0 / gamma)
    benchmark.update_noise([gamma, gamma_phi, gamma_wait0 / gamma])
    infid = average_infidelity(exrec.run().channel_matrix)
    infid_benchmark = benchmark.infidelity()
    print(params, infid, infid_benchmark)
    return infid - infid_benchmark


result = optimize.minimize(g, np.array([4.56243375e+00, 3.25785756e-01, 8.83119261e-04]), method='Nelder-Mead',
                           options={'maxiter': 20})
fitted_params = result.x
print(fitted_params)
