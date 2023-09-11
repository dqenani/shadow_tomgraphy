'''
Clifford Classical Shadows (implemented by @dqenani) builds on the idea of Qiskit Classical Shadows was made to utilize the
 theory of Classical Shadows (originally  designed by @momohuang, https://github.com/momohuang/predicting-quantum-properties) 
 for the quantum state tomography of quantum programs created in qiskit. It extends this idea to include rotations of the
 quantum state by clifford gates instead of just Pauli gates. This expansion allows for more applications of classical shadows
 to problems involving quantum fidelity calculations. It is based off of code provided by Robert Huang 
 https://github.com/momohuang/predicting-quantum-properties and the Pennylane Quantum Computing Framework,
 https://pennylane.ai/qml/demos/tutorial_classical_shadows, as well as
 https://github.com/ryanlevy/shadow-tutorial/blob/main/Tutorial_ShadowQPT_Tomography.ipynb. 
 As with the original Qiskit Classical shadows implementation it integrates well with most quantum programs
  created using the qiskit quantum computing library.
'''

from qiskit import QuantumCircuit, transpile, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import random_clifford

import math
import time
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor
import itertools

from CliffordAssist import shadow_assist

import numpy as np
sd = np.random.seed(666)

'''
    This function takes a qiskit quantum circuit which creates a quantum state, and returns the 'Classical Shadow' of the state
    In this case classical shadow refers to a list of measurements in the computational basis of the rotated state and the a 
    list of the unitaries which rotated the state which in this case are drawn by the 2^n clifford group.
    Args:
        circ_base (Qiskit Quantum Circuit): the quantum circuit that contains the quantum state to be analyzed
        shadow_size (int): the size of the classical shadow to be calculated
        num_qubits (int): the size of the quantum circuit being inputted
    
    Returns:
            shadow(tuple): returns the classical shadow of the quantum state which is a tuple of two lists, one containing the
            measurement results, and the other containing an enumeration of the unitaries that rotated the state
'''
def calc_shadow(circ_base, shadow_size, num_qubits):
    
    unitaries = [random_clifford(num_qubits, sd) for n in range(shadow_size)] #we randomly sample (shadow_size times) from the 2^n clifford gates
        
    bases =[QuantumCircuit(num_qubits) for d in range(shadow_size)] #Creates a list of all the Quantum Circuits to be Parallelized
    
    values = []
    
    for j in range(shadow_size):
        bases[j] = circ_base.copy() # instantiating the array of quantum circuits with the appropriate quantum state given

    with ProcessPoolExecutor() as executor:
        result = executor.map(shadow_assist, unitaries, bases, itertools.repeat(num_qubits))

    values.extend(list(result))
    values = np.array(values)
    
    return (values, unitaries) #returns the shadow as a tuple

'''
    This function takes a list of measurement results and reconstructs the state |b><b| associated with it.
    It first maps the measurement results to the correpsonding quantum state vector, tensors them together,
    and performs an outer product.
    Args:
        b_list (list of int): list of measurement results
    
    Returns:
            state (np.array): returns the matrix containing the state |b><b|
'''
def b_list_to_state(b_list):
    zero = [1,0]
    one = [0,1]

    result = [1]    
    for b in b_list:
        if b == 1:
            st = zero
        elif b == -1:
            st = one
        else:
            raise Exception(f"Invalid Measurement value, not 0 or 1")
        result = np.kron(result,st)
    
    return np.outer(result,result)

'''
    This function takes a classical shadow and using the theory of measurement channel inversion using Clifford gates as
    highlighted by @momohuang constructs a classical shadow (of size 1) in density matrix form. 
    
    Args:
        b_list (List of int): a list containing the measurement results of the quantum state in the computational basis
        obs_list (List ofint): A list containing the enumeration of the unitaries which rotated the state
    
    Returns:
            snapshot (np.array): returns an array containing the classical shadow (of size 1) in density matrix form
'''
def snapshot_state(b_list, clifford):
    num_qubits = len(b_list)
    identity = np.identity(2 ** num_qubits)
    
    U = clifford.to_matrix() #transform the clifford qiskit gate to unitary matrix in order to rotate the state
    state = b_list_to_state(b_list) #transform the measurement results to a classical matrix state
    
    rho_snapshot = (2 ** num_qubits + 1) * (U.conj().T @ state @ U) - identity
    
    return rho_snapshot

'''
    This function takes a classical shadow and using snapshot to construct a classical shadow in density matrix form, 
    performs median of means averaging to return the classical shadow (density matrix form) of appropriate size 
    
    Args:
        shadow (tuple): the classical shadow of the quantum state which is a tuple of two lists, one containing the
            measurement results, and the other containing an enumeration of the unitaries that rotated the state
    
    Returns:
            rho (np.array): returns an array containing the classical shadow (of appropriate size) in density matrix form
'''
def reconstruction(shadow):
    
    num_snapshots, num_qubits = shadow[0].shape
    
    b_lists, obs_lists = shadow
    
    shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype = complex)
    
    for i in range(num_snapshots):
         shadow_rho += snapshot_state(b_lists[i], obs_lists[i]) #average all the snapshots (of size 1) 
        
    return shadow_rho / num_snapshots   #return the classical shadow density matrix of appropriate size

'''
    This function takes an operator and computes the operator norm, which is defined by @momohuang and allows one to
    determine the efficiency and effectiveness of the classical shadow procedure for a given quantum state/ set of observables
    
    Args:
        O (nd.array): An operator which normally will represent a density matrix 
    
    Returns:
            norm (float): returns the norm of the operator
'''
def operator_norm(R):
    return np.sqrt(np.abs(np.trace(R.conj().T@R)))             
