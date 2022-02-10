# Program to manually encrypt and decrypt a 4-bit RSA cryptographic system.
# And then crack it using Shor's algorithm and a quantum simulator.
# Software simulator from IBM Qiskit is used.
# Original author as a Jupyter Notebook: Smaranjit Ghose
# Modified into a Python Script: Thomas Cherickal
# Built a cross-platform UI using Flutter and Dart: Thomas Cherickal


from qiskit import qiskit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, register
from qiskit import execute, Aer

from DecryptRSA import decrypt
from EncryptRSA import encrypt



qasm_sim = qiskit.Aer.get_backend('qasm_simulator')

def period(a,N):
    
    available_qubits = 16 
    r=-1   
    
    if N >= 2**available_qubits:
        print(str(N)+' is too big for IBMQX')
    
    qr = QuantumRegister(available_qubits)   
    cr = ClassicalRegister(available_qubits)
    qc = QuantumCircuit(qr,cr)
    x0 = randint(1, N-1) 
    x_binary = np.zeros(available_qubits, dtype=bool)
    
    for i in range(1, available_qubits + 1):
        bit_state = (N%(2**i)!=0)
        if bit_state:
            N -= 2**(i-1)
        x_binary[available_qubits-i] = bit_state    
    
    for i in range(0,available_qubits):
        if x_binary[available_qubits-i-1]:
            qc.x(qr[i])
    x = x0
    
    while np.logical_or(x != x0, r <= 0):
        r+=1
        qc.measure(qr, cr) 
        for i in range(0,3): 
            qc.x(qr[i])
        qc.cx(qr[2],qr[1])
        qc.cx(qr[1],qr[2])
        qc.cx(qr[2],qr[1])
        qc.cx(qr[1],qr[0])
        qc.cx(qr[0],qr[1])
        qc.cx(qr[1],qr[0])
        qc.cx(qr[3],qr[0])
        qc.cx(qr[0],qr[1])
        qc.cx(qr[1],qr[0])
        
        result = execute(qc,backend = qasm_sim, shots=1024).result()
        counts = result.get_counts()
        
        results = [[],[]]
        for key,value in counts.items(): 
            results[0].append(key)
            results[1].append(int(value))
        s = results[0][np.argmax(np.array(results[1]))]
    return r

def shors_breaker(N):
    N = int(N)
    while True:
        a=randint(0,N-1)
        g=gcd(a,N)
        if g!=1 or N==1:
            return g,N//g
        else:
            r=period(a,N) 
            if r % 2 != 0:
                continue
            elif pow(a,r//2,N)==-1:
                continue
            else:
                p=gcd(pow(a,r//2)+1,N)
                q=gcd(pow(a,r//2)-1,N)
                if p==N or q==N:
                    continue
                return p,q
            
def modular_inverse(a,m): 
    a = a % m; 
    for x in range(1, m) : 
        if ((a * x) % m == 1) : 
            return x 
    return 1


# Passed one command line argument
# 1. cipher_text
def main(argv):
    
    pos = len(argv)

    bit_length =  4
    
    public_k, private_k = generate_keypair(2**bit_length)
    
    plain_txt = argv[]
    
    cipher_txt, cipher_obj = encrypt(plain_txt, public_k)
    
    print("Encrypted message: {}".format(cipher_txt))
    
    print("Decrypted message: {}".format(decrypt(cipher_obj, private_k)))
    
    N_shor = public_k[1]
    
    p,q = shors_breaker(N_shor)

    phi = (p-1) * (q-1)  
    
    d_shor = modular_inverse(public_k[0], phi) 
    
    # # Lets Crack our Cipher Text using Shor's Algorithm
    print('Message Cracked using Shors Algorithm:')
         
    hackedMessage = (decrypt(cipher_obj, (d_shor,N_shor)))
    
    return hackedMessage




if __name__ =='__main__':
    main(sys.argv) 