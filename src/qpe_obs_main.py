import numpy as np
import scipy.linalg as la
import math
import time
import matplotlib.pyplot as plt
from datetime import datetime

from qiskit.visualization import plot_histogram
from qiskit.providers.fake_provider import FakeMontreal

import Ornstein_Uhlenbeck_Classical_Functions as ou
import QPE_Qiskit_Circuit as qpe
import QPE_Zeromode_Solver_Analysis_Function as af
import QPEClassicalShadow as qcs

np.set_printoptions(precision=3, suppress=True, linewidth=150)

def main():
    # Define the system parameters
    nmax = 3
    dx = 0.01
    a = 3
    Γ = 5
    L = 1
    shift = 0

    #Construct the (linear FPE) operators in the Hermite Polynomial basis
    fpe = qcs.create_FPE_operator(a, Γ, L, nmax + 1)
    U, num_query_qubits, dimension = qpe.get_unitary(fpe, add_half = False)
    
    #define the number of qubits needed for QPE circuit based on system size
    num_precision_qubits = 5
    num_qubits = num_precision_qubits + num_query_qubits
    precision_qubits, query_qubits = qpe.get_qubits(num_precision_qubits, num_query_qubits) 
    
    #Create the quantum phase estimation circuit using the unitary associated with the FPE
    circuit = qpe.QuantumCircuit(num_qubits)
    qpe.qpe(circuit, precision_qubits, query_qubits, U, control_unitary = True, with_inverse_barriers = True, measure = False)
    
    #Construct the observables to be calculated, use shadow_bound to determine the size of the shadow necessary
    #within a given error, and calculate the classical shadow
    x = qcs.create_x(L, nmax + 1, 1)
    shadow_size_bound, k = qcs.x_bound(2e-1, x)
    shadow_size_bound = 50000 #override shadow_size_bound to try a large shadow initially
    shadow = qcs.calc_qpe_shadow(circuit, shadow_size_bound, num_qubits, num_precision_qubits, num_query_qubits)

    #calculate the norm of the system in the position basis
    norm = qcs.calc_norm(a, Γ)

    #calculate the expectation value and variance of the operator
    expectation_x = qcs.calc_moment(shadow, x, norm, k)
    variance_x = qcs.calc_cumulant(shadow, x, norm, k)

    print(f"The statistics are calculated for an Orenstein Uhlenbeck Process with the following parameters: a ={a},Γ = {Γ}, L = {L}")
    print(f"expectation of x: {expectation_x}")
    print(f"variance of x: {variance_x}")

if __name__ == "__main__":
    main()
