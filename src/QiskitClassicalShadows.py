'''
Qiskit Classical Shadows (implemented by @dqenani) was made to utilize the theory of Classical Shadows (originally  designed by @momohuang, https://github.com/momohuang/predicting-quantum-properties) for the quantum state tomography of quantum programs created in qiskit. It is based off of code provided by Robert Huang https://github.com/momohuang/predicting-quantum-properties and the Pennylane Quantum Computing Framework, https://pennylane.ai/qml/demos/tutorial_classical_shadows. It takes advantage of parallel processing and the qiskit native gatesets, making it optimized to the qiskit libraries. Additionally it integrates well with most quantum programs created using the qiskit quantum computing library.
'''

import math
from qiskit.primitives import Estimator
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp

import matplotlib.pyplot as plt
import time

from concurrent.futures import ProcessPoolExecutor
import itertools

import numpy as np
np.random.seed(666)

from ShadowAssist import shadow_assist

'''
    This function takes a qiskit quantum circuit which creates a quantum state, and returns the 'Classical Shadow' of the state
    In this case classical shadow refers to a list of measurements in the computational basis of the rotated state and the a 
    list of the unitaries which rotated the state
    Args:
        circ_base (Qiskit Quantum Circuit): the quantum circuit that contains the quantum state to be analyzed
        shadow_size (int): the size of the classical shadow to be calculated
        num_qubits (int): the size of the quantum circuit being inputted
    
    Returns:
            shadow(tuple): returns the classical shadow of the quantum state which is a tuple of two lists, one containing the
            measurement results, and the other containing an enumeration of the unitaries that rotated the state
'''
def calc_shadow(circ_base, shadow_size, num_qubits):
    
    indices = np.random.randint(0,3, size = (shadow_size, num_qubits)) #Here the unitaries that rotate the state are drawn from a random distribution

    values = []

    bases =[QuantumCircuit(num_qubits) for d in range(shadow_size)] #Creates a list of all the Quantum Circuits to be Parallelized
    
    for j in range(shadow_size):
        bases[j] = circ_base.copy() # instantiating the array of quantum circuits with the appropriate quantum state given

    with ProcessPoolExecutor() as executor:  #Parallel Processing the measurements 
        result = executor.map(shadow_assist, indices, bases, itertools.repeat(num_qubits))

    values.extend(list(result))
    values = np.array(values)
    
    return (values, indices) #returns the shadow as a tuple

'''
    This function takes a classical shadow and using the theory of measurement channel inversion using Pauli Gates as
    highlighted by @momohuang constructs a classical shadow (of size 1) in density matrix form. 
    
    Args:
        b_list (List of int): a list containing the measurement results of the quantum state in the computational basis
        obs_list (List ofint): A list containing the enumeration of the unitaries which rotated the state
    
    Returns:
            snapshot (np.array): returns an array containing the classical shadow (of size 1) in density matrix form
'''

def snapshot(b_list, obs_list):
    num_qubits = len(b_list)
    norm = math.sqrt(2) #This defines the norm of a one qubit superposition to be used in defining the classical version of the Hadamard gate
    
    #Classical vector representation of |0> and |1> bit states
    zero_state = np.array([[1,0], [0,0]])
    one_state = np.array([[0,0], [0,1]])
    
    #Define the classical matrix representation for the necessary quantum gates 
    phase_z = np.array([[1,0],[0, 1j]],dtype = complex)
    hadamard = np.array([[1/norm,1/norm],[1/norm,-1/norm]])
    identity = np.array([[1,0],[0, 1]])
    
    unitaries = [hadamard, hadamard @ phase_z , identity] #unitaries corresponding to X, Y, and Z respectively
    
    rho_snapshot = [1]
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state #convert measurement result to appropriate classical bit
        U = unitaries[int(obs_list[i])] #map the the enumerated unitary to its correponding classical counterpart
        
        local_rho = 3 * (U.conj().T @ state @ U) - identity # inversion of the quantum measurement channel
        rho_snapshot = np.kron(rho_snapshot,local_rho)
        
    return rho_snapshot #returns the appropriate density matrix

'''
    This function takes a classical shadow and using snapshot to construct a classical shadow in density matrix form, performs median of means averaging to return the classical shadow (density matrix form) of appropriate size 
    
    Args:
        shadow (tuple): the classical shadow of the quantum state which is a tuple of two lists, one containing the
            measurement results, and the other containing an enumeration of the unitaries that rotated the state
    
    Returns:
            rho (np.array): returns an array containing the classical shadow (of appropriate size) in density matrix form
'''

def reconstruction(shadow):
    
    num_snapshots, num_qubits = shadow[0].shape
    
    b_lists, obs_lists = shadow 
    
    shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype = complex) #create the appropriate Hilbert Space size for density matrix
    
    for i in range(num_snapshots):
         shadow_rho += snapshot(b_lists[i], obs_lists[i]) #average all the snapshots (of size 1) 
        
    return shadow_rho / num_snapshots  #return the classical shadow density matrix of appropriate size

'''
    This function takes an operator and computes the operator norm, which is defined by @momohuang and allows one to
    determine the efficiency and effectiveness of the classical shadow procedure for a given quantum state/ set of observables
    
    Args:
        O (nd.array): An operator which normally will represent a density matrix 
    
    Returns:
            norm (float): returns the norm of the operator
'''
def op_norm(O):
    return np.sqrt(np.abs(np.trace(O.conj().T@O)))             

'''
    This function is meant to interface with algorithimic post processing that rely
    on the counts found at the end of a quantum experiment instead of the density matrix
    of the final state. It does this by artifically creating counts based on the density matrix
    and the size of the shadow used to construct the density matrix.
    
    Args:
        mat (nd.array): An operator representing the density matrix of a quantum state
        size (int): size of the classical shadow used to construct the density matrix
    Returns:
            counts (dict{str: int}): the counts of an experiment that would produce the above quantum state at the end
'''       
def density_matrix_to_counts(mat, size):
    assert mat.shape[0] == mat.shape[1], "Invalid Density Matrix, not square"
    #assert np.trace(mat) == 1, "Invalid Density Matrix, Trace does not equal 1"
    
    counts = {format(i,'b').zfill(int(math.log2(mat.shape[0]))): int(np.real(mat[i, i]) * size) for i in range(mat.shape[0])}
    
    return dict(sorted(counts.items(),key=operator.itemgetter(1),reverse=True))
       