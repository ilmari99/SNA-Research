import re
import os
import argparse
import matplotlib.pyplot as plt

def parse_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # Extract the last val_mae value
        val_mae_matches = re.findall(r'- val_mae: (\d+\.\d+)', content)
        if val_mae_matches:
            last_val_mae = float(val_mae_matches[-4])
        else:
            last_val_mae = None

        # Extract the average score for PentobiNNPlayer
        avg_score_match = re.search(r"Average score.*'PentobiNNPlayer': (\d+\.\d+)", content)
        if avg_score_match:
            avg_score = float(avg_score_match.group(1))
        else:
            avg_score = None

        # Extract the win rate for PentobiNNPlayer
        win_rate_match = re.search(r"Wins.*'PentobiNNPlayer': (\d+\.\d+)", content)
        if win_rate_match:
            win_rate = float(win_rate_match.group(1))
        else:
            win_rate = None

        if any((val is None for val in [last_val_mae, avg_score, win_rate])):
            return None

        return {
            'file': file_path,
            'val_mae': last_val_mae,
            'avg_score': avg_score,
            'win_rate': win_rate
        }

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Given a directory, parse all *.out files
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory containing the *.out files")
    args = parser.parse_args()
    directory = os.path.abspath(args.directory)
    print(f"Directory: ", directory)

    files = [(root, subdirs, files) for root, subdirs, files in os.walk(directory)]
    results = []
    for root, subdirs, files in files:
        for file in files:
            if not file.endswith(".out"):
                continue
            if "BlokusPentobiTestDataset200K-Emb16-3Conv" not in root:
                continue
            file = os.path.join(root, file)
            result = parse_file(file)
            if result:
                results.append(result)
                print(f"Result: {result['file']}: {result['avg_score']}")

    # Print the results
    for result in results:
        print(result)

    # Extract val_mae and win_rate for the scatter plot
    val_mae = [result['val_mae'] for result in results]
    win_rate = [result['avg_score'] for result in results]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(val_mae, win_rate, alpha=0.7)
    plt.xlabel('Validation MAE')
    plt.ylabel('Win Rate')
    plt.title('Scatter plot of Validation MAE vs Win Rate')
    plt.grid(True)
    plt.savefig('val_mae_vs_win_rate.png')
    print('Scatter plot saved as val_mae_vs_win_rate.png')
