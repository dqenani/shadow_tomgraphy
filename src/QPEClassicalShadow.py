'''
QPEClassicalShadow alters QiskitClassicalShadows in order to custom tailor shadow creation to the QPE Circuit.
It does this with the ultimate goal of estimating moments and cumulants of the zeromode of the Orenstein Uhlenbeck Process
calculated in the Hermite Polynomial basis using Quantum Phase Estimation. It includes additional functions for casting 
(position based) observables into the Hermite Polynomial basis and for decomposing these operators into Pauli operator strings
for efficient expectation value calculation using the classical shadow methodology. 
'''

import math
from qiskit.primitives import Estimator
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.providers.fake_provider import FakeMontreal
from qiskit_ibm_runtime import QiskitRuntimeService, Options, Sampler


import matplotlib.pyplot as plt
import time

from concurrent.futures import ProcessPoolExecutor
import itertools

import operator
import numpy as np
np.random.seed(666)

from QPEShadowAssist import shadow_assist

'''
    This function takes a qiskit Quantum Phase quantum circuit which calculate the eigenspectrum of the Fokker-Planck operator in the Hermite Polynomial basis, and returns the 'Classical Shadow' of the query qubits representing the zeromode in that basis. In this case classical shadow refers to a list of measurements in the computational basis of the rotated state and the a 
    list of the unitaries which rotated the state
    Args:
        circ_base (Qiskit Quantum Circuit): the quantum circuit that contains the quantum state to be analyzed
        shadow_size (int): the size of the classical shadow to be calculated
        num_qubits (int): the size of the quantum circuit being inputted
        num_precision_qubits(int):the number of qubits to determine the zeromode eigenvalue (to a certain precision)
        num_query_qubits(int): contains the number of qubits containing the zeromode eigenvector(log(number of basis states))
     
    Returns:
            shadow(tuple): returns the classical shadow of the quantum state which is a tuple of two lists, one containing the
            measurement results, and the other containing an enumeration of the unitaries that rotated the state
'''
def calc_qpe_shadow(circ_base, shadow_size, num_qubits, num_precision_qubits, num_query_qubits, device = None):
    
    assert num_qubits == num_precision_qubits + num_query_qubits, "allocation of qubits is invalid"
    
    indices = np.random.randint(0,3, size = (shadow_size, num_query_qubits)) #Here the unitaries that rotate the state are drawn from a random distribution
    with ProcessPoolExecutor() as executor: #Parallel Processing the measurements
        #if a backend is given set it as the device to be used
        if device != None:
            #the quantum circuits are instantiated with the appropriate quantum state by copying base circuit
            result = executor.map(shadow_assist, indices, itertools.repeat(circ_base.copy()), itertools.repeat(num_qubits), itertools.repeat(num_precision_qubits), itertools.repeat(num_query_qubits), itertools.repeat(device))
        else:
            result = executor.map(shadow_assist, indices, itertools.repeat(circ_base.copy()), itertools.repeat(num_qubits), itertools.repeat(num_precision_qubits), itertools.repeat(num_query_qubits))
    
    result = np.array(list(result))
    mask = result[:, 0] == 1

    return (result[mask, 1:], indices[mask]) #returns the shadow as a tuple

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
    sdg = np.array([[1,0],[0, -1j]],dtype = complex)
    hadamard = np.array([[1/norm,1/norm],[1/norm,-1/norm]])
    identity = np.array([[1,0],[0, 1]])
    
    unitaries = [hadamard, hadamard @ sdg , identity] #unitaries corresponding to X, Y, and Z respectively
    
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
    counts = {format(i,'b').zfill(int(math.log2(mat.shape[0]))): int(np.abs(np.real(mat[i, i])) * size) for i in range(mat.shape[0])}
    
    return dict(sorted(counts.items(),key=operator.itemgetter(1),reverse=True))

'''
    This function is meant calculate local observables (in the form of Pauli Strings)
    using the classical shadow of the system. Due to the local scrambling of the 
    Pauli Group, the observables can be matched to the invidual qubit pauli operators
    that were used to rotate the state. In this way the expectation value of the local
    observable is spliced from the shadow.
    
    Args:
        shadow (tuple): the classical shadow of the quantum state which is a tuple of two lists, one containing the
            measurement results, and the other containing an enumeration of the unitaries that rotated the state
        observable (string): a string containing the observable in the format:"<Qubit Pauli operator acts on><Pauli Operator>..."
    
    Returns:
            expectation_value (float): the expectation value of the observable
'''  
def estimate_observable(shadow, observable, k=10):
    shadow_size, num_qubits = shadow[0].shape
    
    target_obs = []
    target_locs = []
    
    map_name_to_int = {"X": 0, "Y": 1, "Z": 2}
    
    observable = observable.replace(" ", "")
    
    #Here we loop through the passed observable to parse it into two arrays, one containing the observable id
    #The other containing the qubit location of the observable
    for i in range(0, len(observable) - 1, 2):
        
        assert int(observable[i]) in [k for k in range(0, num_qubits)], f"{observable[i]} is an invalid qubit location for observable"
        target_locs.append(int(observable[i]))
        
        assert observable[i + 1] in ['X', 'Y', 'Z'], f"{observable[i]} is not X, Y, or Z"
        target_obs.append(map_name_to_int[observable[i + 1]])

    target_obs = np.array(target_obs)
    target_locs = np.array(target_locs)

    b_lists, obs_lists = shadow
    means = []

    # loop over the splits of the shadow:
    for i in range(0, shadow_size, shadow_size // k):

        # assign the splits temporarily
        b_lists_k, obs_lists_k = (
            b_lists[i: i + shadow_size // k],
            obs_lists[i: i + shadow_size // k],
        )

        # find the exact matches for the observable of interest at the specified locations
        indices = np.all(obs_lists_k[:, target_locs] == target_obs, axis=1)

        # catch the edge case where there is no match in the chunk
        if sum(indices) > 0:
            # take the product and sum
            product = np.prod(b_lists_k[indices][:, target_locs], axis=1)
            means.append(np.sum(product) / sum(indices))
        else:
            means.append(0)
    
    return np.median(means)

'''
    This function determines the size of the shadow necessary to deterimine a set of expectation
    values within a given level of accuracy
    
    Args:
        error (float): the level of accuracy required for expectation value
            measurement results, and the other containing an enumeration of the unitaries that rotated the state
        observables (list<np.array>): a list of arrays containing the matrix form of the observables
        failure_rate(float): precision of the bound calculation
    
    Returns:
            num_shots (int): the required number of measurements to create appropriately size shadow for error rate
''' 
def shadow_bound(error, observables, failure_rate=0.01):
   
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)
    shadow_norm = (
        lambda op: np.linalg.norm(
            op - np.trace(op) / 2 ** int(np.log2(op.shape[0])), ord=np.inf
        )
        ** 2
    )
    N = 34 * max(shadow_norm(o) for o in observables) / error ** 2
    return int(np.ceil(N * K)), int(K)

'''
    This function calculate the matrix equivalent of an observable (meant to interface with Shadow Bound method above)
    
    Args:
        observable (string): a string containing the observable in the format:"<Qubit Pauli operator acts on><Pauli Operator>..."
    
    Returns:
            obs (np.array[float]): array containing the matrix form of the observable
''' 
def obs_to_matrix(observable):
    map_name_to_matrix = {"X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j],[1j, 0]],dtype = complex), "Z": np.array([[1,0],[0, -1]])}
    obs_matrix = [1]
    
    observable = observable.replace(" ", "")
    
    for i in range(0, len(observable) - 1, 2):
        if observable[i + 1] in ["X", "Y", "Z"]: #Check if there is a better way to enter and verify obs
            obs_matrix = np.kron(obs_matrix, map_name_to_matrix[observable[i + 1]])
        else:
            print("Exception, observable was not X, Y, or Z")
    
    return obs_matrix

'''
    This function take the matrix representation of an observable (hermitatian matrix) in the truncated Hermite polynomial basis
    and decomposes it into Pauli strings
    
    Args:
        mat (np.array): matrix representation of observable
        
    Returns:
             decomposition (tuple(np.array[float],np.array[string])): tuple containing the decomposed Pauli strings and their corresponding weights in the decomposition
''' 
def decompose_operator(mat):
    decomp, coefficients = SparsePauliOp.from_operator(mat).paulis, SparsePauliOp.from_operator(mat).coeffs
    operator_strings = []

    for op in decomp:
        op_str = str(op)
        result = ''.join([str(index) + o for index, o in enumerate(op_str) if o != "I"])
        operator_strings.append(result)
    
    return (coefficients.real, np.array(operator_strings))

'''
    This function determines the size of the shadow necessary to deterimine a set of expectation
    values within a given level of accuracy for observables made up of the position operator  
    
    Args:
        error (float): the level of accuracy required for expectation value
            measurement results, and the other containing an enumeration of the unitaries that rotated the state
        observables (list<np.array>): a list of arrays containing the matrix form of the observables
        failure_rate(float): precision of the bound calculation
    
    Returns:
            num_shots (int): the required number of measurements to create appropriately size shadow for error rate
''' 
def x_bound(error, x_observable, failure_rate=0.01):
    coeffs, list_of_observables = decompose_operator(x_observable)
    shadow_size_bound, k = shadow_bound(error, observables=[obs_to_matrix(o) for o in list_of_observables])
    return shadow_size_bound, k
'''
    This function  takes the string representing an observable and transforms it into a Sparse Pauli Operator Qiskit Object
     (meant to interface with Qiskit estimator to calculate exact expectation values of a circuit using Qiskit)
    
    Args:
        observable (string): a string containing the observable in the format:"<Qubit Pauli operator acts on><Pauli Operator>..."
   
    Returns:
            obs (SparsePauliOp): Qiskit object containing observable information
''' 
def obs_to_op(obs, system_size):
    op = ["I" for i in range(system_size)]
    
    for i in range(0, len(obs) - 1, 2):
        assert (obs[i + 1] in ["X", "Y", "Z"]), "observable was not X, Y, or Z"
        assert (int(obs[i]) in [k for k in range(0, system_size)]), "Invalid qubit index"
        op[int(obs[i])] = obs[i + 1]

    return SparsePauliOp(Pauli("".join(op)))

'''
    This function returns the matrix representation of the raising operator acting on the truncated Hermite Polynomial basis
    of a given maximum size (a_dag|n> = sqrt(n + 1)|n+1>).
    
    Args:
        num_hermite_polynomials (int): largest hermite polynomial basis state considered in expansion
    
    Returns:
            a_dag (np.array[float]): numpy array representing the raising operator
''' 
def raising_operator(num_hermite_polynomials):
    a_dag = np.zeros((num_hermite_polynomials, num_hermite_polynomials))
    
    for i in range(num_hermite_polynomials):
        for j in range(num_hermite_polynomials):
            if (i == j + 1):
                a_dag[i][j] = math.sqrt(j + 1)
    
    return a_dag

'''
    This function returns the matrix representation of the lowering operator acting on the truncated Hermite Polynomial basis
    of a given maximum size (a|n> = sqrt(n)|n-1>).
    
    Args:
        num_hermite_polynomials (int): largest hermite polynomial basis state considered in expansion
    
    
    Returns:
            a_dag (np.array[float]): numpy array representing the lowering operator
''' 
def lowering_operator(num_hermite_polynomials):
    a = np.zeros((num_hermite_polynomials, num_hermite_polynomials))
    
    for i in range(num_hermite_polynomials):
        for j in range(num_hermite_polynomials):
            if (i == j - 1):
                a[i][j] = math.sqrt(j)
    
    return a
    
'''
    This function returns the matrix representation of the FPE operator for the one-dimensional Orenstein Uhlenbeck Process
    in the truncated Hermite Polynomial basis of a given maximum size ( L_FPE(P) = d/dx(a*x*P)  + Γ * d^2P/dx^2).
    
    Args:
        a (float): potential strength for Orenstein-Uhlenbeck equation
        Γ (float): additive white noise strength for Orenstein-Uhlenbeck equation
        L (float): the length scale used
        num_hermite_polynomials (int): largest hermite polynomial basis state considered in expansion

    Returns:
            fpe (np.array[float]): numpy array representing the fpe operator
''' 
def create_FPE_operator(a, Γ, L, num_hermite_polynomials):
    num_hermite = num_hermite_polynomials + 1 #To avoid errors with early truncation calculate (matrices) using an additional basis state and then truncate at the end
    fpe = np.zeros((num_hermite, num_hermite))
    term1 = (a/2)*((lowering_operator(num_hermite) - raising_operator(num_hermite))@(lowering_operator(num_hermite) + raising_operator(num_hermite)))
    term2 =((Γ*(L**2))/ 2)*((lowering_operator(num_hermite) - raising_operator(num_hermite))@(lowering_operator(num_hermite) - raising_operator(num_hermite)))
    fpe = term1 + term2
    return fpe[:num_hermite_polynomials, :num_hermite_polynomials]

'''
    This function returns the matrix representation of the position operator of a given dimension in the truncated Hermite 
    Polynomial basis of a given maximum size (x = 1/(Lsqrt(2))(a + a_dag)).
    
    Args:
        L (float): the length scale used
        num_hermite_polynomials (int): largest hermite polynomial basis state considered in expansion
        exponent (int): the dimension of the operator

    Returns:
            x^n (np.array[float]): numpy array representing the position operator of a given dimension
''' 
def create_x(L, num_hermite_polynomials, exponent = 1):
    num_hermite = num_hermite_polynomials + 1
    x = np.zeros((num_hermite, num_hermite))
    x = 1/(L * math.sqrt(2)) * (raising_operator(num_hermite) + lowering_operator(num_hermite))
    result = x
    for i in range(exponent - 1):
        result = result @ x
    return result[:num_hermite_polynomials, :num_hermite_polynomials]

'''
    This function calculates the position basis norm of the zeromode PDF.
    
    Args:
        a (float): potential strength for Orenstein-Uhlenbeck equation
        Γ (float): additive white noise strength for Orenstein-Uhlenbeck equation
        
    Returns:
            norm (float): the norm of the zeromode
''' 
def calc_norm(a, Γ):
    d = a / (2 * Γ)
    return math.sqrt(d/(math.pi)) 

'''
    This function calculates the moment (expectation value) of an observable using the zeromode in the Hermite Polynomial
    basis, taking advantage of classical shadows
    
    Args:
        shadow (tuple): the classical shadow for state capturing the zeromode of the system
        observable(np.array[float]): the matrix representation (in the Hermite polynomial basis) of the observable
        degree(int): the degree of the observable being calculated, i.e. x = degree 1, x^2 = degree 2
        norm(float): the norm of the zeromode PDF in the postion basis, used to scale and mitigate the effects of the P^2 is used to calculate expectation values 
    
    Returns:
             moment (float): the moment of the observable
''' 
def calc_moment(shadow, observable, norm, degree = 1, k = None):
    coeffs, ops = decompose_operator(observable)
    moment = 0
    for c, o in zip(coeffs, ops):
        if o == "":
            moment += c 
        else:
            moment += c * (estimate_observable(shadow, o, k) if k is not None else estimate_observable(shadow, o)) 
    #since the the moment was calculated using P^2 not P, where P is the PDF of the system, the moment must be normalized
    scale = (2**(degree//2)*math.sqrt(2))/(norm)
    #moment =  scale * moment
    return moment

'''
    This function calculates the second cumulant (the variance) of an observable using the zeromode in the Hermite Polynomial
    basis, taking advantage of classical shadows
    
    Args:
        shadow (tuple): the classical shadow for state capturing the zeromode of the system
        observable(np.array[float]): the matrix representation (in the Hermite polynomial basis) of the observable
        norm(float): the norm of the zeromode PDF in the postion basis, used to scale and mitigate the effects of the P^2 is used to calculate expectation values 
    
    Returns:
             cumulant (float): the second cumulant of the observable
''' 
def calc_cumulant(shadow, observable, norm, degree = 1, k = None):
    observable_square = observable @ observable
    
    m_obs = calc_moment(shadow, observable, norm, k = None) #enter degree as parameter
    m_obs_square = calc_moment(shadow, observable_square, norm, k = None) #enter degree * 2 as parameter 
    
    return (m_obs_square - ((m_obs)*(m_obs)))