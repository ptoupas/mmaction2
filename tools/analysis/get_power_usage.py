import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average GPU power consumption')
    parser.add_argument('--filename', type=str, help='the log file from the nvidia-smi report')
    args = parser.parse_args()

    with open(args.filename) as f:
        pow_results = [float(line.split()[0]) for line in f if 'power' not in line]
    pow_results = pow_results[10:]
    print(f'Average Power Usage: {sum(pow_results)/len(pow_results):.4} W')