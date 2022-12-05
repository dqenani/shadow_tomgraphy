#This python script creates a Majumdar Gosh Model to be used in conjunction with the Simulation Script
#These two scripts simulate a quantum system to be used in the calculation of expectation values using Shadow Tomography
import numpy as np
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

def task(cur_measurement, system_size):
    #Create the quantum circuit with the number of qubits and classical bits equivalent to the quantum system size
        circuit = QuantumCircuit(system_size)
        simulator = Aer.get_backend('aer_simulator')#QasmSimulator()

        #create the quantum system
        #Here we implement the demo system described in the predicting many properties sample code : 5 consecutive qubit
        #singlet states
        for i in range(0, system_size, 2): #Was previously for i in range(0, 10, 2)
            circuit.h(i)
            circuit.x(i+1)
            circuit.z(i)
            circuit.cnot(i,i+1)
            circuit.x(i)
            circuit.x(i+1)

        #Remove all the whitespace from the line
        cur_measurement = cur_measurement.replace(' ','').strip()

        #Measure each qubit in the appropriate basis as given by the measurement scheme
        counter = 0
        for j in cur_measurement: 
            print(j)      
            if (j == 'Z'):
                #already in computational basis 
                continue
            elif (j == 'X'):
                circuit.h(counter)
            elif (j == 'Y'):
                circuit.h(counter)
                circuit.s(counter)
            else:
                raise Exception(f"Invalid Measurement Basis: input {j} is not a member of the Pauli Bases")

            counter = counter + 1

        #Measure the circuit
        circuit.measure_all()

        #Run the simulation of the circuit
        compiled_circuit = transpile(circuit, simulator)
        job = simulator.run(compiled_circuit, shots=1)
        result = job.result().get_counts(compiled_circuit)

        #loop through the individual measurement results for each qubit and add the basis and result to the current
        #measurement output
        line = ""
        counter = 0
        for res in result:
            for r in res:
                if (r == '0'):
                    r = '1'
                elif (r == '1'):
                    r = '-1'
                line = line + cur_measurement[counter] + " " + r + " "
                counter = counter + 1
        return line    