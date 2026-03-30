import os
import warnings
import argparse
import torch
import numpy as np
import pennylane as qml


def parse():
    parser = argparse.ArgumentParser('Evaluate effective channel')
    parser.add_argument('--code', type=str, default='perfect',
                        choices=['perfect', 'bitflip',
                                 'yflip_varqec-5_layer-1', 'yflip_varqec-5_layer-2',
                                 'bit_varqec-5_layer-1', 'bit_varqec-5_layer-2',
                                 'bit_varqec-3_layer-1', 'bit_varqec-3_layer-2',
                                 'adep_varqec-4_layer-1', 'adep_varqec-5_layer-1'],
                        help='The QEC code (encoding and recover) to load.')
    parser.add_argument('--p', type=float, required=True,
                        help='Total strength of initial Pauli noise channel.')
    parser.add_argument('--px', type=float, required=True,
                        help='Proportion of Pauli-X component of noise (`px` + `pz` !<= 1; `py` := 1 - `px` - `pz`).')
    parser.add_argument('--pz', type=float, required=True,
                        help='Proportion of Pauli-Z component of noise (`px` + `pz` !<= 1; `py` := 1 - `px` - `pz`).')
    args = parser.parse_args()
    if args.px + args.pz > 1:
        raise ValueError('The relative error proportions cannot sum to more than one.')
    args.py = 1 - args.px - args.pz
    print(f'Pauli noise channel before correction:     '
          f'px/p={args.px:.2f}, py/p={args.py:.2f}, pz/p={args.pz:.2f} | p={args.p:.3g}')
    return args


def load_code(code: str):
    # load encoding
    path_encoding = os.path.join('codes', code, f'encoding.qasm')
    if not os.path.isfile(path_encoding):
        raise FileNotFoundError(f'No encoding ansatz found at {path_encoding}.')
    with open(path_encoding, 'r') as f:
        qasm_encoding = f.read()
    # load recovery
    path_recovery = os.path.join('codes', code, f'recovery.qasm')
    if not os.path.isfile(path_recovery):
        raise FileNotFoundError(f'No recovery ansatz found at {path_recovery}.')
    with open(path_recovery, 'r') as f:
        qasm_recovery = f.read()
    # extract code parameters
    data_qubit = qasm_encoding.split('\n')[0][2:].split(',')[0]
    ancilla_qubits = qasm_encoding.split('\n')[1][2:].split(',')
    recovery_qubits = qasm_recovery.split('\n')[2][2:].split(',')
    if code in ['perfect', 'bitflip']:
        print(f'Applying [[{len(ancilla_qubits) + 1},1]] `{code}` code ...')
    else:
        print(f'Applying (({len(ancilla_qubits) + 1},2)) VarQEC code ...')
    return qasm_encoding, qasm_recovery, data_qubit, ancilla_qubits, recovery_qubits


def construct_kraus_matrices(p: float, px_p: float, py_p: float, pz_p: float):
    # construct matrices
    k0 = np.sqrt(1 - p + qml.math.eps) * qml.math.convert_like(np.eye(2, dtype=complex), p)
    k1 = np.sqrt(p * px_p + qml.math.eps) * qml.math.convert_like(
        np.array([[0, 1], [1, 0]], dtype=complex), p)
    k2 = np.sqrt(p * py_p + qml.math.eps) * qml.math.convert_like(
        np.array([[0, -1j], [1j, 0]], dtype=complex), p)
    k3 = np.sqrt(p * pz_p + qml.math.eps) * qml.math.convert_like(
        np.array([[1, 0], [0, -1]], dtype=complex), p)
    return [k0, k1, k2, k3]


def circuit(qasm_encoding: str, qasm_recovery: str,
            data_qubit: str, ancilla_qubits: list[str], recovery_qubits: list[str],
            kraus_matrices: list = None):
    # two-design states
    parameters_two_design = torch.transpose(torch.tensor([
        [0.0, 0.0, 0.0],  # state |0>
        [np.pi, 0.0, np.pi],  # state |1>
        [np.pi / 2, 0.0, np.pi],  # state |+>
        [np.pi / 2, -np.pi, -np.pi],  # state |->
        [np.pi / 2, np.pi / 2, -np.pi],  # state |+i>
        [np.pi / 2, -np.pi / 2, -np.pi]  # state |-i>
    ]), 0, 1)
    qml.U3(*parameters_two_design, wires=[data_qubit])

    # encoding
    qml.from_qasm3(qasm_encoding, wire_map={w: w for w in [data_qubit] + ancilla_qubits})()

    # noise channel
    if kraus_matrices is not None:
        for w in [data_qubit] + ancilla_qubits:
            qml.QubitChannel(kraus_matrices, wires=w)

    # recovery
    qml.from_qasm3(qasm_recovery, wire_map={w: w for w in [data_qubit] + ancilla_qubits + recovery_qubits})()

    # decoding
    qml.adjoint(qml.from_qasm3(qasm_encoding, wire_map={w: w for w in [data_qubit] + ancilla_qubits}))()

    # measure only data qubit
    return qml.density_matrix(wires=[data_qubit])  # noqa


def project_probability_simplex(px: float, py: float, pz: float):
    """
    Projection as described in J. Duchi et al., "Efficient projections onto the l1-ball for learning in high dimensions"
    """
    # initial checks of already valid
    if px >= 0 and py >= 0 and pz >= 0 and px + py + pz <= 1:
        return px, py, pz

    # first step:  clip non-zero components (equivalent to direct projection, where clipping happens inherently)
    px, py, pz = np.maximum(px, 0.0), np.maximum(py, 0.0), np.maximum(pz, 0.0)
    # check if valid now
    if px + py + pz <= 1:
        return px, py, pz

    # second step: project to ({p >= 0, sum p_i = 1})
    v = np.array([px, py, pz])
    # sort in descending order
    u = np.sort(v)[::-1]
    # compute cumulative sums
    ccsv = np.cumsum(u)
    # find index of last component that will be positive (Lemma 2)
    j =  np.arange(1, len(v) + 1)
    cond = u * j > (ccsv - 1)
    rho = np.nonzero(cond)[0][-1]
    # compute threshold (Eq. 5)
    theta = (ccsv[rho] - 1.0) / (rho + 1.0)
    # construct projected vector (Eq. 6)
    w = np.maximum(v - theta, 0.0)
    return w[0], w[1], w[2]



def evaluate_effective_channel(args):
    qasm_encoding, qasm_recovery, data_qubit, ancilla_qubits, recovery_qubits = load_code(args.code)

    # set up pennylane device
    device = qml.device('default.mixed', wires=[data_qubit] + ancilla_qubits + recovery_qubits)
    qnode = qml.QNode(circuit, device)

    # run evaluation
    with torch.no_grad():
        with warnings.catch_warnings(action="ignore", category=UserWarning):  # filter harmless but annoying PyTorch warning
            # noise-free
            target = qnode(qasm_encoding, qasm_recovery, data_qubit, ancilla_qubits, recovery_qubits)
            # target noise channel
            prediction = qnode(qasm_encoding, qasm_recovery, data_qubit, ancilla_qubits, recovery_qubits,
                               construct_kraus_matrices(p=args.p, px_p=args.px, py_p=args.py, pz_p=args.pz))
    # compute pairwise fidelities between target and prediction
    fidelities = qml.math.fidelity(target, prediction).numpy()  # noqa

    # least-squares fit of Pauli channel (Bloch-picture)
    nz = fidelities[0] + fidelities[1] - 1  # |0> and |1>
    nx = fidelities[2] + fidelities[3] - 1  # |+> and |->
    ny = fidelities[4] + fidelities[5] - 1  # |+i> and |-i>

    # convert to Kraus representation
    px_hat, py_hat, pz_hat = (1 - ny - nz + nx) / 4, (1 - nx - nz + ny) / 4, (1 - nx - ny + nz) / 4

    # project to (Euclidean-) closest physical Pauli channel
    px, py, pz = project_probability_simplex(px=px_hat, py=py_hat, pz=pz_hat)

    # construct and print effective channel
    p = px + py + pz
    print(f'Effective pauli channel after correction:  '
          f'px/p={px / p:.2f}, py/p={py / p:.2f}, pz/p={pz / p:.2f} | p={p:.3g}')


if __name__ == '__main__':
    evaluate_effective_channel(parse())
