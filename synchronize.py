from TPM import TPM
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Synchronize two TPMs')
parser.add_argument('k', type=int, help='the number of hidden units')
parser.add_argument('n', type=int, help='the number of units connected to the hidden units')
parser.add_argument('l', type=int, help='maximum weight value: {+l, ..., -3, -2, -1, 0, 1, 2, 3, ..., -l}')
parser.add_argument('update_rule', type=str, help='the update rule to use for the TPMs')
parser.add_argument('--verbose', action='store_true', help='print data about the running state of the program')
args = parser.parse_args()

k = args.k
n = args.n
l = args.l
update_rule = args.update_rule

if args.verbose:
    verbose = True
else:
    verbose = False

def score(tpm1, tpm2):
    matching_elements = np.sum(tpm1.W == tpm2.W)
    total_elements = tpm1.W.size  
    proportion_of_matches = matching_elements / total_elements
    scaled_score = 1 + 99 * proportion_of_matches
    return scaled_score

def random_number_generator(l, k, n):
    return np.random.randint(-l, l + 1, (k, n))

if verbose:
    print('Creating TPM1 with k = {}, n = {}, l = {}'.format(args.k, args.n, args.l))

tpm1 = TPM(args.k, args.n, args.l)

if verbose:
    print('Creating TPM2 with k = {}, n = {}, l = {}'.format(args.k, args.n, args.l))

tpm2 = TPM(args.k, args.n, args.l)

if verbose:
    print('Creating man in the middle machine with k = {}, n = {}, l = {}'.format(args.k, args.n, args.l))

evil_tpm = TPM(args.k, args.n, args.l)

if verbose:
    print('Synchronizing TPM1 and TPM2 using the {} update rule'.format(args.update_rule))

sync = False
epoch = 0

while not sync:
    X = random_number_generator(l, k, n)

    output1 = tpm1.output(X)
    output2 = tpm2.output(X)
    evil_output = evil_tpm.output(X)

    tpm1.update(output2, update_rule)
    tpm2.update(output1, update_rule)

    if output1 == output2 == evil_output:
        evil_tpm.update(output1, update_rule)

    score_tpm = score(tpm1, tpm2)
    epoch += 1

    if verbose:
        print('Score: {}'.format(score_tpm))
        print('Epoch: {}'.format(epoch))

    if score_tpm == 100:
        sync = True

if verbose:
    print('TPM1 and TPM2 have synchronized after {} epochs'.format(epoch))

    print('Final score: {}'.format(score_tpm))

evil_score = score(tpm1, evil_tpm)

if evil_score == 100:
    print("COMPROMISED")
else:
    print("SECURE")

print('\n\n--------------------------------------\n\n')
print("Public key: {}".format(X))
print('\n\n--------------------------------------\n\n')
print("Private key: {}".format(tpm1.W))

# Write the public key to a file
with open('public_key.txt', 'w') as f:
    f.write(str(X))

# Write the private key to a file
with open('private_key.txt', 'w') as f:
    f.write(str(tpm1.W))

print(tpm1.output(X))
