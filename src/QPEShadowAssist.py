from qiskit.primitives import Estimator
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.providers.fake_provider import FakeMontreal
from qiskit.quantum_info.operators import Operator, Pauli
import numpy as np

default_device = Aer.get_backend('aer_simulator')
map_result = {"0": 1, "1": -1}

'''
    This function acts as a helper function for the parallelization of calc_shadow in QPEClassicalShadows. It takes the quantum circuit defining QPE and unitaries to
     rotate the state, performs the measurement, and returns a list of the results.
     
     Args:
         indices (List of int): contains a list of the enumerated unitaries to rotate the state (Pauli gates: X, Y, Z)
         circ (QuantumCircuit): the quantum circuit which contains the quantum state to rotate and measure
         num_qubits(int): the size of the quantum system
         num_precision_qubits(int):the number of qubits to determine the zeromode eigenvalue (to a certain precision)
         num_query_qubits(int): contains the number of qubits containing the zeromode eigenvector(log(number of basis states))
     
     Returns:
         vals(List of int): returns a list of the measurement results                      
 '''
def shadow_assist(indices, circ, num_qubits, num_precision_qubits, num_query_qubits, device = default_device):
    vals = np.zeros(num_query_qubits + 1)

    count = 0
    for index in indices:
        if (index == 0): #Pauli X
            circ.h(num_precision_qubits + count)
        elif (index == 1): #Pauli Y
            circ.sdg(num_precision_qubits + count)
            circ.h(num_precision_qubits + count)
        elif (index == 2): #Pauli Z
            circ.id(num_precision_qubits + count)
        else:
            raise Exception(f"Invalid Measurement Basis: input on {count} qubit is not a member of the Pauli Bases")
        
        count = count + 1
        
    circ.measure_all()
                    
    simulator = device
    compiled_circuit = transpile(circ, simulator)
    job = simulator.run(compiled_circuit, shots=1)
    result = job.result().get_counts(compiled_circuit)
    parsed_result = list(result)[0][::-1]
    
    is_zeromode = 1 if all(parsed_result[j] == "0" for j in range(num_precision_qubits)) else 0
    
    truncated_result = parsed_result[num_precision_qubits:]
    vals = [is_zeromode] + [map_result[r] for r in truncated_result]

    return vals