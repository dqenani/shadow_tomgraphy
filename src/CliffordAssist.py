'''
    This script simply contains a helper function for the parallelization of calc_shadow
'''
from qiskit.primitives import Estimator
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info.operators import Operator, Pauli
import numpy as np

'''
    This function takes a quantum circuit and a clifford unitary gate to
    rotate the state. It proceeds to rotate thes state, perform the measurement, and returns a list of the results.
    
    Args:
        unitary (clifford gate): contains a qiskit gate of a randomly selected clifford gate (made up of its stablizer and destablizer)
        circ (QuantumCircuit): the quantum circuit which contains the quantum state to rotate and measure
        num_qubits(int): the size of the quantum system
        
    
    Returns:
        vals(List of int): returns a list of the measurement results                      
'''
def shadow_assist(unitary, circuit, num_qubits):
    vals = np.zeros(num_qubits)
    circ = circuit 
    circ = circ.compose(unitary.to_circuit()) #Here we append the clifford gate to the circuit containing the quantum state
        
    circ.measure_all(inplace = True)
                    
    simulator = Aer.get_backend('aer_simulator')
    compiled_circuit = transpile(circ, simulator)
    job = simulator.run(compiled_circuit, shots=1)
    result = job.result().get_counts(compiled_circuit)
    c = 0
    for res in result:
        start = len(res) - num_qubits
        res = res[start:]
        c= 0
        for r in res:
            if (r == '0'):
                vals[c] = 1 
            elif (r == '1'):
                vals[c] = -1 
            c = c + 1  
    return list(vals)