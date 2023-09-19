from qiskit.primitives import Estimator
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.providers.aer import QasmSimulator 
from qiskit.providers.fake_provider import FakeMontreal
from qiskit.quantum_info.operators import Operator, Pauli
import numpy as np

#global variables to be used in the parallel processed function below
default_device = Aer.get_backend('aer_simulator')
map_result = {"0": 1, "1": -1}


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

def shadow_assist(indices, circ, num_qubits, device = default_device):
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
    circ.measure_all(inplace = True)
                    
    simulator = device
    compiled_circuit = transpile(circ, simulator)
    job = simulator.run(compiled_circuit, shots=1)
    result = job.result().get_counts(compiled_circuit)

    vals = [map_result[r] for r in list(result)[0]]
 
    return vals