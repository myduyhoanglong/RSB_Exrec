KNILL = 0
KNILL_NO_PREP_NOISE = 1
HYBRID = 2

DIRECT = 0
MAXIMUM_LIKELIHOOD = 1

FAST = 0
SDP = 1
FRONT = 2

ALPHA_MAX = 9
DIM = int(ALPHA_MAX ** 2 + max(36, ALPHA_MAX * 6))

EXREC_LOG = 1
OPTIMIZE_LOG = 2

NO_ERROR = 0  # 1(.)1
LOSS = 1  # a(.)a^{\dagger}
DEPHASE_OD = 2  # n(.) + (.)n
LOSS_SQ = 3  # a^2 (.) a^{\dagger}^2
DEPHASE_SQ_OD = 4  # n^2(.) + (.)n^2
LOSS_DEPHASE = 5  # na(.)a^{\dagger} + a(.)a^{\dagger}n
PHASE = 6  # e^{i(\pi/N^2)n}(.)e^{-i(\pi/N^2)n}
PHASE_SQ = 7  # e^{i(\2pi/N^2)n}(.)e^{-i(\2pi/N^2)n}
TWO_DEPHASE_OD = 8  # n^2(.) + (.)n^2 + 2n(.)n
DEPHASE_PHASE = 9  # ne^{i(\pi/N^2)n}(.)e^{-i(\pi/N^2)n} + e^{i(\pi/N^2)n}(.)e^{-i(\pi/N^2)n}n
LOSS_PHASE = 10  # e^{i(\2pi/N^2)n}a(.)a^{\dagger}e^{-i(\2pi/N^2)n}
DEPHASE_LOSS = 11  # an(.)a^{\dagger} + a(.)na^{\dagger}
DEPHASE = 12  # n(.)n

init_pairs = [(2, 5), (4, 10), (6, 15), (8, 20)]