from qiskit.primitives import Estimator
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info.operators import Operator, Pauli
import numpy as np

'''
    This function acts as a helper function for the parallelization of calc_shadow. It takes quantum circuit and unitaries to
    rotate the state, performs the measurement, and returns a list of the results.
    
    Args:
        indices (List of int): contains a list of the enumerated unitaries to rotate the state (Pauli gates: X, Y, Z)
        circ (QuantumCircuit): the quantum circuit which contains the quantum state to rotate and measure
        num_qubits(int): the size of the quantum system
        
    
    Returns:
        vals(List of int): returns a list of the measurement results                      
'''

def shadow_assist(indices, circ, num_qubits):
    vals = np.zeros(num_qubits)

    count = 0
    for index in indices:
        if (index == 0): #Pauli X
            circ.h(count)
        elif (index == 1): #Pauli Y
            circ.sdg(count)
            circ.h(count)
        elif (index == 2): #Pauli Z
            circ.id(count)
        else:
            raise Exception(f"Invalid Measurement Basis: input on {count} qubit is not a member of the Pauli Bases")
        
        count = count + 1
    circ.measure_all()
                    
    simulator = Aer.get_backend('aer_simulator') 
    compiled_circuit = transpile(circ, simulator)
    job = simulator.run(compiled_circuit, shots=1)
    result = job.result().get_counts(compiled_circuit)

    for res in result: # we parse the dictionary of measurement results ({measurement: number of times measured}) to get result
        start = len(res) - num_qubits
        res = res[start:]
        c = 0
        for r in res:
            if (r == '0'):
                vals[c] = 1 
            elif (r == '1'):
                vals[c] = -1 
            c = c + 1  
    return list(vals)