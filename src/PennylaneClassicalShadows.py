import pennylane as qml
import pennylane.numpy as np 
import matplotlib.pyplot as plt
import time

np.random.seed(666)

def calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits):
    
    outcomes = np.zeros((shadow_size, num_qubits))
    
    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]
    unitary_ids = np.random.randint(0,3, size = (shadow_size, num_qubits))
    for ns in range(shadow_size):
        
        obs = [unitary_ensemble[int(unitary_ids[ns, i])](i) for i in range(num_qubits)]
        outcomes[ns,:] = circuit_template(params, observables = obs)
        
    return (outcomes, unitary_ids)

def snapshot_state(b_list, obs_list):
    num_qubits = len(b_list)
    
    zero_state = np.array([[1,0], [0,0]])
    one_state = np.array([[0,0], [0,1]])
    
    phase_z = np.array([[1,0],[0,-1j]],dtype = complex)
    hadamard = qml.matrix(qml.Hadamard(0))
    identity = qml.matrix(qml.Identity(0))
    
    unitaries = [hadamard, hadamard @ phase_z, identity]
    
    rho_snapshot = [1]
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = unitaries[int(obs_list[i])]
        
        local_rho = 3 * (U.conj().T @ state @ U) - identity
        rho_snapshot = np.kron(rho_snapshot,local_rho)
        
    return rho_snapshot

def shadow_state_reconstruction(shadow):
    
    num_snapshots, num_qubits = shadow[0].shape
    
    b_lists, obs_lists = shadow
    
    shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype = complex)
    
    for i in range(num_snapshots):
         shadow_rho += snapshot_state(b_lists[i], obs_lists[i])
        
    return shadow_rho / num_snapshots   

def operator_norm(R):
    return np.sqrt(np.abs(np.trace(R.conjugate().transpose()@R)))

def estimate_shadow_obervable(shadow, observable, k=10):
    shadow_size, num_qubits = shadow[0].shape

    # convert Pennylane observables to indices
    map_name_to_int = {"PauliX": 0, "PauliY": 1, "PauliZ": 2}
    if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        target_obs, target_locs = np.array(
            [map_name_to_int[observable.name]]
        ), np.array([observable.wires[0]])
    else:
        target_obs, target_locs = np.array(
            [map_name_to_int[o.name] for o in observable.obs]
        ), np.array([o.wires[0] for o in observable.obs])

    # classical values
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