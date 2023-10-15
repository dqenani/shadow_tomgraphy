'''
The purpose of this python script is to run QPE (as implemented by @dwei7)
with classical shadow tomography on the supercomputer OSCAR (at Brown University).
The use of classical shadows is to recreate the density matrix of the quantum
state at the end of QPE before post-processing. The aim is to explore how this
technique may improve the efficiency of the algorithm and reduce the amount of 
total measurements needed to accurately produce the results (mitigating the effect
of Born's rule).
'''
import numpy as np
import scipy.linalg as la
import math
import time
from datetime import datetime

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile
from qiskit.visualization import plot_histogram
from qiskit.providers.fake_provider import FakeMontreal

import Ornstein_Uhlenbeck_Classical_Functions as ou
import QPE_Qiskit_Circuit as qpe
import QPE_Zeromode_Solver_Analysis_Function as af
import QiskitClassicalShadows as qcs

#GlOBAL VARIABLES
ADD_HALF = False
NUM_QUERY_QUBITS = 2
NUM_PRECISION_QUBITS = 5
num_qubits = NUM_PRECISION_QUBITS + NUM_QUERY_QUBITS

def create_system(nmax, dx, a, Γ, c, L, shift):
    x = np.linspace(-5, 5, int((5+5)/dx))
    y = ou.perturbed_ornstein_uhlenbeck(x, a = a, Γ = Γ, c = c, shift = shift)
    return (x, y)

def classical_solution(nmax, x, y, L, dx, a, Γ, c):
    cache_projection = ou.integrate_eigenvector(x, y, nmax, L)
    x_projection, y_projection = ou.reconstruct_eigenvector(cache_projection)
    op_nonhermitian = ou.create_operator_perturbed(nmax, L, a, c, Γ)
    only_even = True
    cache_diagonalization = ou.find_zeromode(op_nonhermitian, nmax, x, dx, L, which = "single", only_even = only_even)
    x_diagonalization, y_diagonalization = ou.reconstruct_eigenvector(cache_diagonalization, only_even = only_even)
    return cache_diagonalization

def  classical_zeromode(U):
    A, P = la.eig(U)
    A_real = np.real(A)
    if ADD_HALF:
        target = np.exp(1j * np.pi)
    else:
        target = np.exp(1j * 0)
    diff = A - target
    r_diff = np.real(diff * np.conjugate(diff))
    index = np.where(r_diff == np.amin(r_diff))[0][0]
    eigenvalue = A[index]
    zeromode_classic = P[:, index] ## zeromode_classic is the zeromode of U from classical diagonalization
    zeromode_classic = np.real(ou.normalize_eigenvector(zeromode_classic))
    zeromode_classic = np.reshape(zeromode_classic, (zeromode_classic.size, 1))
    return (eigenvalue, zeromode_classic)

def create_fpe_operator(cache_diagonalization):
    matrix = cache_diagonalization["operator"]
    np.random.seed(6)  
    matrix = ou.generate_matrix([0, np.pi/2, np.pi, 3*np.pi/2])
    U, num_query_qubits, dimension = qpe.get_unitary(matrix, add_half = ADD_HALF)
    return (U, num_query_qubits, dimension)

def create_quantum_circuits(U):
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)

    circ_o = QuantumCircuit(qr, cr) #Normal Circuit
    circ_s = QuantumCircuit(num_qubits) #Shadows Circuit
    q_precision_qubits, q_query_qubits = qpe.get_qubits(NUM_PRECISION_QUBITS, NUM_QUERY_QUBITS)
    s_precision_qubits, s_query_qubits = qpe.get_qubits(NUM_PRECISION_QUBITS, NUM_QUERY_QUBITS)
    qpe.qpe(circ_o, q_precision_qubits, q_query_qubits, U, control_unitary = True, with_inverse_barriers = True, measure = True)
    #Setup a circuit to be ready for classical shadow tomography
    qpe.qpe(circ_s, s_precision_qubits, s_query_qubits, U, control_unitary = True, with_inverse_barriers = True, measure = False)
    return (circ_o, circ_s, q_precision_qubits, q_query_qubits, s_precision_qubits, s_query_qubits)

def quantum_zeromode(circ_o, num_snapshots, precision_qubits, query_qubits, backend = Aer.get_backend('aer_simulator')):
    t_circ = transpile(circ_o, backend)
    task = backend.run(t_circ, shots = num_snapshots)
    result = task.result()
    result_cache2 = af.post_process_results(result, precision_qubits)
    zeromode_qpe2, n_hit2 = af.find_eigenvector(result_cache2, len(precision_qubits), len(query_qubits), make_even = False)
    return (zeromode_qpe2, n_hit2 )

def shadow_zeromode(circ_s, num_snapshots, precision_qubits, query_qubits, backend = Aer.get_backend('aer_simulator')):
    shadow = qcs.calc_shadow(circ_s, num_snapshots, num_qubits, backend)        
    dmat = qcs.reconstruction(shadow)
    counts = qcs.density_matrix_to_counts(dmat, num_snapshots)
    result_cache = af.post_process_shadows(counts, precision_qubits)
    zeromode_qpe, n_hit = af.find_eigenvector(result_cache, len(precision_qubits), len(query_qubits), make_even = False)
    return (zeromode_qpe, n_hit)

def shadow_comparison(circ_s, s_precision_qubits, s_query_qubits, zeromode_classic, number_runs, snapshot_range, backend = Aer.get_backend('aer_simulator')):
    shadow_distances =  []

    for i in range(number_runs):
        d = ""
        for j, num_snapshots in enumerate(snapshot_range):
            s_zeromode, s_n_hits = shadow_zeromode(circ_s.copy(), num_snapshots, s_precision_qubits, s_query_qubits, backend)
            print(f"shadow zeromode ({num_snapshots} shots): {s_zeromode}")
            d = d + f"{np.dot(np.transpose(s_zeromode), np.abs(zeromode_classic))[0]}, " 
        shadow_distances.append(d[:-2])
    return shadow_distances

def quantum_comparison(circ_o, q_precision_qubits, q_query_qubits, zeromode_classic, number_runs, snapshot_range, backend = Aer.get_backend('aer_simulator')):
    quantum_distances =  [] 

    for i in range(number_runs):
        d = ""
        for j, num_snapshots in enumerate(snapshot_range):
            q_zeromode, s_n_hits = quantum_zeromode(circ_o.copy(), num_snapshots, q_precision_qubits, q_query_qubits, backend)
            print(f"quantum zeromode ({num_snapshots} shots): {q_zeromode}")
            d = d + f"{np.dot(np.transpose(q_zeromode), np.abs(zeromode_classic))[0]}, " 
        quantum_distances.append(d[:-2])
    return quantum_distances

def format_time(t):
    t = int(t)
    hours = int(t // 3600)
    t = t % 3600
    minutes = int (t // 60)
    seconds = t % 60
    return f"{hours} hours: {minutes} minutes: {seconds} seconds"

def main():
    # Creating the System
    nmax = 15
    dx = 0.01
    a = 3
    Γ = 5
    c = 1
    L = 1
    shift = 0
    x, y = create_system(nmax, dx, a,  Γ, c, L, shift)

    #create classical operator
    cache_diagonalization = classical_solution(nmax, x, y, L, dx, a, Γ, c)

    #Create Quantum Operator (Unitary Matrix)
    U, num_query_qubits, dimension = create_fpe_operator(cache_diagonalization)
    
    #Find Classical Zeromode
    eigenvalue_classic, zeromode_classic = classical_zeromode(U)

    #Create quantum circuit
    circ_o, circ_s, q_precision_qubits, q_query_qubits, s_precision_qubits, s_query_qubits = create_quantum_circuits(U)

    number_runs = 2
    snapshot_range = [10000, 50000, 100000]

    print("Backend Used: Fake Montreal")
    print(f"\n Snapshot Range = {snapshot_range} \n")

    start = time.time()
    s1 = time.time()
    s_distances = shadow_comparison(circ_s, s_precision_qubits, s_query_qubits, zeromode_classic, number_runs, snapshot_range, backend = FakeMontreal())
    e1 = time.time()

    s2 = time.time()
    q_distances = quantum_comparison(circ_o, q_precision_qubits, q_query_qubits, zeromode_classic, number_runs, snapshot_range, backend = FakeMontreal())
    e2 = time.time()
    end = time.time()

    shadow_time = s1 - e1
    quantum_time = s2 - e2
    run_time = start - end

    print(f"With Shadow Results:")
    for i in range(number_runs):
        print(f"{s_distances[i]}")

    print(f"\n Without Shadow Results:")
    for j in range(number_runs):
        print(f"{q_distances[j]}")
    

    shadow_time = format_time(shadow_time)
    print(f"\nShadow Run Time = {shadow_time}")
    quantum_time = format_time(quantum_time)
    print(f"Without Shadow Run Time = {quantum_time}")
    run_time = format_time(run_time)
    print(f"Total Run Time = {run_time}")
    
if __name__ == "__main__":
    main()
