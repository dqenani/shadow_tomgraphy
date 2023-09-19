from qiskit.primitives import Estimator
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.fake_provider import FakeMontreal
from qiskit.quantum_info.operators import Operator, Pauli
import numpy as np

default_device = Aer.get_backend('aer_simulator')
map_result = {"0": 1, "1": -1}

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