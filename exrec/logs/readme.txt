Data is collected for HYBRID and KNILL schemes with the following parameters. 
Common parameters: recovery=MAXIMUM_LIKELIHOOD, decoder=TRANSPOSE
Fixed parameters:
	KNILL: M=N
	HYBRID: M=1, alpha_anc = ALPHA_MAX, offset_anc = 0
Optimized parameters:
	KNILL: alpha_data, alpha_anc, offset_data, offset_anc, eta (waiting time)
	HYBRID: alpha_data, offset_data, eta

Extended gadget setup: leading EC + waiting + trailing EC
There is double counting in the simulation, because it's hard to separate the contribution from locations of one EC. 

Scan parameters:
N = 2, 3, 4
gamma_phi = 0, 10**(-4) to 10**(-2) with step 0.2 in power
gamma = 0, around the boundary

Initial parameters:
KNILL: (alpha_data,alpha_anc) = (3,8), (5,8)
	 (offset_data, offset_anc) = (-0.1, -0.1) to (-0.3, -0.3) (depending on N)
	  eta = 5, 10, 15, 20
HYBRID: alpha_data = 3, 5, 7
	  offset_data = 0, -0.1, -0.2, -0.3 (depending on N)
	  eta = 5, 10, 15, 20